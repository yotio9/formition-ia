import base64
import io
import os
import re
import unicodedata

from huggingface_hub import InferenceClient

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
_IMAGE_ACTION = re.compile(
    r"\b(gener(?:e|er|ate)|cre(?:e|er|ate)|dessin(?:e|er)|draw|make|"
    r"fabriqu(?:e|er)|produi(?:s|re)|imagin(?:e|er)|fai(?:s|re))\b"
)
_IMAGE_SUBJECT = re.compile(
    r"\b(image|illustration|dessin|photo|portrait|picture|artwork)\b"
)


def est_demande_image(texte: str) -> bool:
    normalise = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", texte.casefold())
        if not unicodedata.combining(caractere)
    )
    return bool(_IMAGE_ACTION.search(normalise) and _IMAGE_SUBJECT.search(normalise))


def generer_image(prompt: str) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("LLM_MODEL_IMAGE")
    if not token:
        raise ValueError(
            "Le token Hugging Face n'est pas configuré. "
            "Définis HF_TOKEN dans le fichier .env."
        )

    client = InferenceClient(provider="hf-inference", token=token)
    image = client.text_to_image(prompt=prompt, model=IMAGE_MODEL)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{image_base64}"
