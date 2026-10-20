from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from formation_ia.chat.models import generer_reponse
from formation_ia.chat.routes import router as chat_router
from rag import obtenir_modele_embeddings
from utils import embedding_functions as cosinus

app = FastAPI()
app.include_router(chat_router)


@app.get("/", include_in_schema=False)
async def accueil():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/llm")
async def appeler_llm(
    system: str,
    user: str,
    temperature: float = 0.2,
):
    try:
        contenu = generer_reponse(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )

        return {
            "reponse": contenu
        }
    except Exception as e:
        print(f"Erreur lors de l'appel à l'LLM : {e}")
        return {"reponse": "Erreur lors de l'appel à l'LLM ", "details": str(e)}


@app.post("/emb")
async def embeddings(phrases: list[str]):
    if not phrases:
        return {"erreur": "La liste phrases ne peut pas être vide."}

    modele_embeddings = obtenir_modele_embeddings()
    vecteurs = modele_embeddings.encode(phrases)
    similarites = [
        [cosinus(a, b) for b in vecteurs]
        for a in vecteurs
    ]
    return {
        "phrases": phrases,
        "embeddings": vecteurs.tolist(),
        "similarites": similarites,
    }
        
