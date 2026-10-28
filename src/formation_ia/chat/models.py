import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)


def get_available_models() -> list[str]:
    configured = [
        os.getenv("LLM_MODEL"),
        os.getenv("LLM_MODEL_2"),
        os.getenv("LLM_MODEL_3"),
        os.getenv("LLM_MODEL_4"),
    ]
    preferred = list(
        dict.fromkeys(
            model.strip()
            for model in configured
            if model and model.strip()
        )
    )
    if preferred:
        return preferred

    reponse = client.models.list()
    available = list(
        dict.fromkeys(
            model.id.strip()
            for model in reponse.data
            if model.id and model.id.strip()
        )
    )
    return available


def resolve_model_name(
    model: str | None = None,
    available_models: list[str] | None = None,
) -> str:
    models = (
        available_models
        if available_models is not None
        else get_available_models()
    )
    if model and model.strip():
        requested = model.strip()
        if requested in models:
            return requested
        raise ValueError(f"Le modèle demandé n'est pas disponible : {requested}.")
    if not models:
        raise ValueError("Le fournisseur n'a renvoyé aucun modèle disponible.")
    return models[0]


def generer_reponse(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    model: str | None = None,
    available_models: list[str] | None = None,
) -> str | None:
    model_name = resolve_model_name(model, available_models)

    reponse = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return reponse.choices[0].message.content
