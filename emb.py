from sentence_transformers import SentenceTransformer

from utils import cosinus

modele_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


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