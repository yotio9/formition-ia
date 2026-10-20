import hashlib
import io
import os
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError

_DATABASE_PATH = Path(
    os.getenv(
        "RAG_DB_PATH",
        str(Path(__file__).resolve().parent / ".rag" / "index.sqlite3"),
    )
)
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_INITIALIZED_PATH: Path | None = None
_CACHE_LOCK = threading.RLock()
_CACHE_REVISION: int | None = None
_CACHE_ROWS: list[sqlite3.Row] = []
_CACHE_VECTORS = np.empty((0, 0), dtype=np.float32)


@lru_cache(maxsize=1)
def obtenir_modele_embeddings():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def _connexion() -> sqlite3.Connection:
    global _SCHEMA_INITIALIZED_PATH

    _DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connexion = sqlite3.connect(_DATABASE_PATH, timeout=30)
    connexion.row_factory = sqlite3.Row
    connexion.execute("PRAGMA foreign_keys = ON")
    try:
        with _SCHEMA_LOCK:
            if _SCHEMA_INITIALIZED_PATH != _DATABASE_PATH:
                connexion.execute("PRAGMA journal_mode = WAL")
                connexion.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        page_count INTEGER NOT NULL,
                        chunk_count INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS chunks (
                        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        page INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        PRIMARY KEY (document_id, chunk_index)
                    );
                    CREATE INDEX IF NOT EXISTS chunks_document_page
                        ON chunks (document_id, page, chunk_index);
                    CREATE TABLE IF NOT EXISTS index_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        revision INTEGER NOT NULL
                    );
                    INSERT OR IGNORE INTO index_state (id, revision) VALUES (1, 0);
                    """
                )
                _SCHEMA_INITIALIZED_PATH = _DATABASE_PATH
    except Exception:
        connexion.close()
        raise
    return connexion


def decouper(texte: str, taille: int = 220, chevauchement: int = 40):
    if taille <= 0 or chevauchement < 0 or chevauchement >= taille:
        raise ValueError("Le chevauchement doit être positif et inférieur à la taille.")

    mots = texte.split()
    pas = taille - chevauchement
    for debut in range(0, len(mots), pas):
        morceau = " ".join(mots[debut : debut + taille])
        if morceau:
            yield morceau


def indexer_pdf(contenu_pdf: bytes, nom: str) -> dict[str, Any]:
    if not contenu_pdf.startswith(b"%PDF-"):
        raise ValueError("Le fichier envoyé n'est pas un PDF valide.")

    try:
        lecteur = PdfReader(io.BytesIO(contenu_pdf))
        if lecteur.is_encrypted:
            raise ValueError("Le PDF est protégé par un mot de passe.")

        morceaux: list[tuple[int, str]] = []
        for numero_page, page in enumerate(lecteur.pages, start=1):
            texte = page.extract_text() or ""
            morceaux.extend(
                (numero_page, morceau)
                for morceau in decouper(texte)
            )
    except PdfReadError as error:
        raise ValueError("Impossible de lire ce PDF. Vérifie que le fichier n'est pas endommagé.") from error

    if not morceaux:
        raise ValueError(
            "Aucun texte n'a été trouvé dans ce PDF. Les PDF scannés nécessitent une OCR."
        )

    vecteurs = np.asarray(
        obtenir_modele_embeddings().encode(
            [morceau for _, morceau in morceaux],
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vecteurs.ndim != 2 or vecteurs.shape[0] != len(morceaux):
        raise RuntimeError("Le modèle d'embeddings a renvoyé un résultat inattendu.")

    identifiant = hashlib.sha256(contenu_pdf).hexdigest()
    lignes = [
        (
            identifiant,
            index,
            page,
            texte,
            np.ascontiguousarray(vecteurs[index]).tobytes(),
        )
        for index, (page, texte) in enumerate(morceaux)
    ]
    connexion = _connexion()
    try:
        with connexion:
            connexion.execute(
                """
                INSERT INTO documents (id, name, page_count, chunk_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    indexed_at = CURRENT_TIMESTAMP,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count
                """,
                (identifiant, nom, len(lecteur.pages), len(lignes)),
            )
            connexion.execute("DELETE FROM chunks WHERE document_id = ?", (identifiant,))
            connexion.executemany(
                """
                INSERT INTO chunks
                    (document_id, chunk_index, page, content, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                lignes,
            )
            connexion.execute("UPDATE index_state SET revision = revision + 1 WHERE id = 1")
    finally:
        connexion.close()

    return {
        "document": nom,
        "pages": len(lecteur.pages),
        "chunks": len(lignes),
    }


def lister_documents() -> list[dict[str, Any]]:
    connexion = _connexion()
    try:
        return [
            dict(ligne)
            for ligne in connexion.execute(
                """
                SELECT name AS document, page_count AS pages,
                       chunk_count AS chunks, indexed_at
                FROM documents
                ORDER BY indexed_at DESC, name
                """
            )
        ]
    finally:
        connexion.close()


def _charger_index() -> tuple[list[sqlite3.Row], np.ndarray]:
    global _CACHE_REVISION, _CACHE_ROWS, _CACHE_VECTORS

    with _CACHE_LOCK:
        connexion = _connexion()
        try:
            revision = connexion.execute(
                "SELECT revision FROM index_state WHERE id = 1"
            ).fetchone()["revision"]
            if revision == _CACHE_REVISION:
                return _CACHE_ROWS, _CACHE_VECTORS

            lignes = connexion.execute(
                """
                SELECT chunks.content, chunks.page, documents.name AS document,
                       chunks.embedding
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                ORDER BY documents.name, chunks.page, chunks.chunk_index
                """
            ).fetchall()
        finally:
            connexion.close()

        if lignes:
            vecteurs = np.vstack(
                [
                    np.frombuffer(ligne["embedding"], dtype=np.float32)
                    for ligne in lignes
                ]
            )
        else:
            vecteurs = np.empty((0, 0), dtype=np.float32)

        _CACHE_REVISION = revision
        _CACHE_ROWS = lignes
        _CACHE_VECTORS = vecteurs
        return _CACHE_ROWS, _CACHE_VECTORS


def retrouver(question: str, k: int = 5) -> list[dict[str, Any]]:
    if not question.strip():
        raise ValueError("La question ne peut pas être vide.")
    if k < 1:
        raise ValueError("Le nombre d'extraits doit être supérieur à zéro.")

    lignes, vecteurs = _charger_index()
    if not lignes:
        return []

    question_vecteur = np.asarray(
        obtenir_modele_embeddings().encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )[0]
    if question_vecteur.shape[0] != vecteurs.shape[1]:
        raise RuntimeError("La dimension de l'embedding ne correspond pas à l'index.")

    scores = vecteurs @ question_vecteur
    nombre = min(k, len(lignes))
    meilleurs = np.argpartition(scores, -nombre)[-nombre:]
    meilleurs = meilleurs[np.argsort(scores[meilleurs])[::-1]]
    return [
        {
            "document": lignes[index]["document"],
            "page": lignes[index]["page"],
            "texte": lignes[index]["content"],
            "score": float(scores[index]),
        }
        for index in meilleurs
    ]
