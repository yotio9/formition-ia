import os
#from	pypdf	import	PdfReader

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

LLM_MODEL = os.getenv("LLM_MODEL")
modele = SentenceTransformer('all-MiniLM-L12-v2')


@app.post("/llm")
async def appeler_llm(
    system: str,
    user: str,
    temperature: float = 0.2,
):
    try:
        reponse = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        temperature=temperature,
        )

        return {
            "reponse": reponse.choices[0].message.content
        }
    except Exception as e:
        print(f"Erreur lors de l'appel à l'LLM : {e}")
        return {"reponse": "Erreur lors de l'appel à l'LLM ", "details": str(e)}


modele_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def cosinus(a, b):
    normes = np.linalg.norm(a) * np.linalg.norm(b)
    if normes == 0:
        return 0.0
    return float(np.dot(a, b) / normes)


@app.post("/emb")
async def embeddings(phrases: list[str]):
    if not phrases:
        return {"erreur": "La liste phrases ne peut pas être vide."}

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
        

