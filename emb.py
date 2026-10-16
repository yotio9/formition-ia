import numpy as np
from sentence_transformers import SentenceTransformer

modele_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def cosinus(a, b):
    normes = np.linalg.norm(a) * np.linalg.norm(b)
    if normes == 0:
        return 0.0
    return float(np.dot(a, b) / normes)


def embeddings(phrases: list[str]):
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


if __name__ == "__main__":
    phrases_test = [
        "Bonjour, comment vas-tu ?",
        "Salut, comment ça va ?",
        "Le temps est ensoleillé aujourd'hui.",
    ]
    resultat = embeddings(phrases_test)
    for phrase, similarites in zip(
        resultat["phrases"], resultat["similarites"], strict=True
    ):
        print(f"{phrase}: {similarites}")