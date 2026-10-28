import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from formation_ia.chat import models, routes


class ModelTests(unittest.TestCase):
    def test_liste_les_modeles_configures_dans_env(self) -> None:
        page = SimpleNamespace(
            data=[
                SimpleNamespace(id="provider-model-a"),
                SimpleNamespace(id="provider-model-b"),
                SimpleNamespace(id="provider-model-a"),
            ]
        )
        with (
            patch.object(models.client.models, "list", return_value=page),
            patch.dict(
                "os.environ",
                {
                    "LLM_MODEL": "configured-model-a",
                    "LLM_MODEL_2": "provider-model-b",
                    "LLM_MODEL_3": " configured-model-c ",
                    "LLM_MODEL_4": "",
                },
                clear=False,
            ),
        ):
            self.assertEqual(
                models.get_available_models(),
                ["configured-model-a", "provider-model-b", "configured-model-c"],
            )
            models.client.models.list.assert_not_called()

    def test_liste_les_modeles_du_fournisseur_si_env_est_vide(self) -> None:
        page = SimpleNamespace(
            data=[
                SimpleNamespace(id="provider-model-a"),
                SimpleNamespace(id="provider-model-b"),
                SimpleNamespace(id="provider-model-a"),
            ]
        )
        with (
            patch.object(models.client.models, "list", return_value=page),
            patch.dict(
                "os.environ",
                {
                    "LLM_MODEL": "",
                    "LLM_MODEL_2": "",
                    "LLM_MODEL_3": "",
                    "LLM_MODEL_4": "",
                },
                clear=False,
            ),
        ):
            self.assertEqual(
                models.get_available_models(),
                ["provider-model-a", "provider-model-b"],
            )

    def test_chat_transmet_le_modele_choisi_au_client(self) -> None:
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)
        available_models = ["provider-model-a", "provider-model-b"]

        with (
            patch.object(routes, "get_available_models", return_value=available_models),
            patch.object(routes, "lister_documents", return_value=[]),
            patch.object(routes, "generer_reponse", return_value="Réponse du modèle") as generate,
        ):
            response = client.post(
                "/chat",
                json={
                    "model": "provider-model-b",
                    "messages": [{"role": "user", "content": "Bonjour"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reponse"], "Réponse du modèle")
        self.assertEqual(generate.call_args.kwargs["model"], "provider-model-b")
        self.assertEqual(
            generate.call_args.kwargs["available_models"],
            available_models,
        )

    def test_chat_refuse_un_modele_absent_du_fournisseur(self) -> None:
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)

        with patch.object(
            routes,
            "get_available_models",
            return_value=["provider-model-a"],
        ):
            response = client.post(
                "/chat",
                json={
                    "model": "unknown-model",
                    "messages": [{"role": "user", "content": "Bonjour"}],
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("n'est pas disponible", response.json()["detail"])

    def test_appel_utilise_le_modele_choisi(self) -> None:
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Réponse"))]
        )
        with patch.object(
            models.client.chat.completions,
            "create",
            return_value=completion,
        ) as create:
            response = models.generer_reponse(
                [{"role": "user", "content": "Bonjour"}],
                model="configured-model-b",
                available_models=["configured-model-a", "configured-model-b"],
            )

        self.assertEqual(response, "Réponse")
        self.assertEqual(create.call_args.kwargs["model"], "configured-model-b")


if __name__ == "__main__":
    unittest.main()
