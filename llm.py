import os
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from formation_ia.chat.routes import router as chat_router
from formation_ia.chat.images import est_demande_image, generer_image
from rag import obtenir_modele_embeddings
from utils import embedding_functions as cosinus
from openai import OpenAI
from dotenv import load_dotenv

app = FastAPI()
app.include_router(chat_router)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

load_dotenv()
URL = os.getenv("LLM_URL") or os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")
KEY = os.getenv("LLM_API_KEY")

client = OpenAI(
    base_url=URL,
    api_key=KEY
)

class QueryRequest(BaseModel):
    question: str


@app.get("/", include_in_schema=False)
async def accueil(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/llm")
async def appeler_llm(resquest: QueryRequest):
    if est_demande_image(resquest.question):
        try:
            image = generer_image(resquest.question)
        except ValueError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Impossible de générer l'image. Vérifie le token Hugging Face et réessaie.",
            ) from error
        return {"reponse": "Voici l’image générée.", "image": image}

    context_rag = await rags([resquest.question])  # Ajusté en liste pour correspondre aux attentes de /emb

    system_prompt = (
        "Tu es un assistant IA. Tu dois répondre à la question de l'utilisateur.\n"
        "Consignes :\n"
        "1. Si le 'Contexte' ci-dessous contient la réponse, utilise-le en priorité pour répondre.\n"
        "2. Si le contexte est vide ou inutile, ou si c'est une question générale (ex: code, histoire, salutations), "
        "ignore le contexte et utilise tes propres connaissances générales pour répondre."
    )

    try:
        reponse = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexte RAG:\n{context_rag}\n\nQuestion: {resquest.question}"}
            ],
            temperature=0.3
        )
        # CORRECTION 3 : La bonne syntaxe OpenAI est 'choices[0]' (avec un 's')
        return {"reponse": reponse.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM: {str(e)}")


@app.post("/emb")
async def rags(phrases: list[str]):
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
