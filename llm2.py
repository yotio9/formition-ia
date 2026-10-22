from fastapi import FastAPI
import os
import openai
from dotenv  import load_dotenv
import json
from pydantic import BaseModel
import requests

load_dotenv()
URL=os.getenv("LLM_URL")
MODEL=os.getenv("LLM_MODEL")
KEY=os.getenv("LLM_API_KEY")










def appeler_llm(system: str, user: str, temperature: float = 0.2) -> str:
    reponse = requests.post(
        URL,
        headers={"Authorization": f"Bearer {KEY}"},
        json={
            "model": MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=60,  # un modèle peut être lent
    )
    reponse.raise_for_status()  # erreur claire si clé fausse (401) ou quota (429)
    return reponse.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(appeler_llm(
        "Tu réponds en français, en deux phrases maximum.",
        "Explique ce qu'est un token.",
    ))