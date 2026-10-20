import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import rag


class FauxPage:
    def __init__(self, texte: str) -> None:
        self.texte = texte

    def extract_text(self) -> str:
        return self.texte


class FauxLecteurPDF:
    is_encrypted = False

    def __init__(self, source: BytesIO) -> None:
        if source.read(5) != b"%PDF-":
            raise AssertionError("L'en-tête du faux PDF doit être valide.")
        self.pages = [FauxPage("alpha sujet"), FauxPage("beta sujet")]


class FauxModele:
    def encode(self, phrases: list[str], **options: object):
        if options.get("normalize_embeddings") is not True:
            raise AssertionError("Les embeddings doivent être normalisés.")
        return rag.np.asarray(
            [[1.0, 0.0] if "alpha" in phrase else [0.0, 1.0] for phrase in phrases],
            dtype=rag.np.float32,
        )


class RagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repertoire_temporaire = tempfile.TemporaryDirectory()
        rag._DATABASE_PATH = Path(self.repertoire_temporaire.name) / "index.sqlite3"
        rag._CACHE_REVISION = None
        rag._CACHE_ROWS = []
        rag._CACHE_VECTORS = rag.np.empty((0, 0), dtype=rag.np.float32)

    def tearDown(self) -> None:
        self.repertoire_temporaire.cleanup()

    def test_decouper_garde_un_chevauchement(self) -> None:
        self.assertEqual(
            list(rag.decouper("un deux trois quatre", taille=3, chevauchement=1)),
            ["un deux trois", "trois quatre"],
        )

    def test_indexation_et_recherche_retournent_la_page_source(self) -> None:
        with (
            patch.object(rag, "PdfReader", FauxLecteurPDF),
            patch.object(rag, "obtenir_modele_embeddings", FauxModele),
        ):
            resultat = rag.indexer_pdf(b"%PDF-1.4 contenu", "inf-dexe.pdf")
            extraits = rag.retrouver("alpha")

        self.assertEqual(resultat, {"document": "inf-dexe.pdf", "pages": 2, "chunks": 2})
        self.assertEqual(extraits[0]["document"], "inf-dexe.pdf")
        self.assertEqual(extraits[0]["page"], 1)
        self.assertEqual(extraits[0]["texte"], "alpha sujet")

    def test_un_pdf_reindexe_ne_cree_pas_de_doublon(self) -> None:
        with (
            patch.object(rag, "PdfReader", FauxLecteurPDF),
            patch.object(rag, "obtenir_modele_embeddings", FauxModele),
        ):
            rag.indexer_pdf(b"%PDF-1.4 contenu", "inf-dexe.pdf")
            rag.indexer_pdf(b"%PDF-1.4 contenu", "renomme.pdf")

        self.assertEqual(len(rag.lister_documents()), 1)
        self.assertEqual(rag.lister_documents()[0]["document"], "renomme.pdf")

    def test_refuse_un_fichier_qui_n_est_pas_un_pdf(self) -> None:
        with self.assertRaisesRegex(ValueError, "n'est pas un PDF valide"):
            rag.indexer_pdf(b"texte sans entete PDF", "document.txt")


if __name__ == "__main__":
    unittest.main()
