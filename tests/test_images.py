import base64
import os
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from formation_ia.chat.images import est_demande_image, generer_image
from formation_ia.chat.routes import router
from fastapi import FastAPI


class ImageGenerationTests(unittest.TestCase):
    def test_reconnait_une_demande_de_generation_en_francais(self) -> None:
        self.assertTrue(est_demande_image("Génère une image d'un chat roux"))

    def test_ne_confond_pas_une_question_generale_avec_une_demande(self) -> None:
        self.assertFalse(est_demande_image("Qu'est-ce qu'une image numérique ?"))

    def test_generer_image_encode_le_png_retourne(self) -> None:
        image = Image.new("RGB", (1, 1), color="red")
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), patch(
            "formation_ia.chat.images.InferenceClient"
        ) as client_class:
            client_class.return_value.text_to_image.return_value = image

            resultat = generer_image("un chat roux")

        self.assertTrue(resultat.startswith("data:image/png;base64,"))
        png_bytes = base64.b64decode(resultat.split(",", 1)[1])
        self.assertEqual(png_bytes[:8], b"\x89PNG\r\n\x1a\n")
        client_class.assert_called_once_with(provider="hf-inference", token="test-token")
        client_class.return_value.text_to_image.assert_called_once_with(
            prompt="un chat roux",
            model="black-forest-labs/FLUX.1-schnell",
        )

    def test_chat_retourne_l_image_sans_appeler_le_modele_de_texte(self) -> None:
        app = FastAPI()
        app.include_router(router)
        with patch(
            "formation_ia.chat.routes.generer_image",
            return_value="data:image/png;base64,cG5n",
        ) as generate:
            response = TestClient(app).post(
                "/chat",
                json={
                    "messages": [
                        {"role": "user", "content": "Génère une image de forêt"}
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["image"], "data:image/png;base64,cG5n")
        self.assertEqual(response.json()["reponse"], "Voici l’image générée.")
        generate.assert_called_once_with("Génère une image de forêt")

    def test_token_absent_produit_une_erreur_de_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "token Hugging Face"):
                generer_image("un chat roux")


if __name__ == "__main__":
    unittest.main()
