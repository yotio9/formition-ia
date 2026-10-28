# Mon IA

Interface de chat en français avec recherche dans des PDF (RAG) et modèle configuré dans ce projet.

## Démarrer

Configure les variables suivantes dans un fichier `.env` à la racine du projet :

```env
LLM_BASE_URL=https://ton-serveur-ia/v1
LLM_API_KEY=ta-cle-api
LLM_MODEL=nom-du-modele-par-defaut
HF_TOKEN=ton-token-hugging-face
```

Renseigne `LLM_MODEL` et, si nécessaire, `LLM_MODEL_2`, `LLM_MODEL_3` ou
`LLM_MODEL_4` avec les identifiants de modèles acceptés par ton fournisseur.
Ces identifiants alimentent la liste déroulante et le modèle choisi est envoyé
avec chaque question. Si aucune de ces variables n'est renseignée, l'interface
récupère la liste du fournisseur. Tu peux choisir un modèle dans l'interface,
voir le modèle actif et actualiser la liste.

Le token Hugging Face sert à la génération d'images. Garde les vraies valeurs
uniquement dans `.env` : ce fichier est ignoré par Git. Tu peux partir de
`.env.example` et remplacer ses valeurs d'exemple.

Puis démarre le serveur :

```bash
uv run uvicorn llm:app --reload
```

Ouvre ensuite <http://127.0.0.1:8000>. La page garde les conversations dans le
navigateur et envoie l'historique au modèle pour conserver le contexte. Les
routes `/llm` et `/emb` existantes restent disponibles.

## Poser des questions sur un PDF

Dans l'interface, clique sur **Ajouter un PDF** et sélectionne `inf-dexe.pdf`
(ou un autre PDF). Le serveur extrait le texte, le découpe en extraits avec
chevauchement, calcule les embeddings en lots puis conserve l'index local dans
`.rag/index.sqlite3`. Le nombre de PDF indexés et leur liste sont affichés dans
la barre latérale ; le bouton rouge **Retirer** supprime le PDF et ses extraits
de l'index. Les fichiers PDF ne sont pas conservés par le serveur.
Les questions sont recherchées dans cet index et les réponses affichent leurs
documents et pages sources. Le modèle d'embeddings est téléchargé lors de sa
première utilisation. Les PDF composés uniquement de scans nécessitent une OCR,
qui n'est pas incluse.

L'index peut être déplacé en définissant `RAG_DB_PATH` avant le démarrage.
L'envoi est limité à 20 Mo par PDF.

## Organisation du chat

- `src/formation_ia/chat/routes.py` : routes HTTP du chat et validation métier.
- `src/formation_ia/chat/schemas.py` : schémas des messages et des requêtes/réponses.
- `src/formation_ia/chat/models.py` : configuration et appels au modèle IA.
- `rag.py` : indexation persistante et recherche vectorielle des extraits.
- `llm.py` : création de l'application FastAPI et conservation des routes existantes.
