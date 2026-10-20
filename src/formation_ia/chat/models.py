import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL")

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


def generer_reponse(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
) -> str | None:
    if not LLM_MODEL:
        raise ValueError("La variable LLM_MODEL n'est pas configurée.")

    reponse = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return reponse.choices[0].message.content
