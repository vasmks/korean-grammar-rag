from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fastapi import Response
from fastapi.testclient import TestClient

from app.main import add_topik_preview_urls, app, health, topik_preview_service


class ApiContractTests(TestCase):
    def test_required_routes_exist(self):
        paths = {route.path for route in app.routes}
        self.assertTrue(
            {
                "/",
                "/health",
                "/ask",
                "/topik/preview/{exam_number}/{question_start}/{question_end}",
            }.issubset(paths)
        )

    def test_health_shape(self):
        rag = SimpleNamespace(entries=[1, 2], topik=SimpleNamespace(units=[1]))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(rag=rag)))
        self.assertEqual(
            health(request),
            {"status": "ok", "dictionary_entries": 2, "topik_documents": 1},
        )

    def test_preview_url_is_added(self):
        result = {
            "topik_occurrences": [
                {"exam_number": 60, "question_start": 13, "question_end": 13}
            ]
        }
        add_topik_preview_urls(result)
        self.assertEqual(
            result["topik_occurrences"][0]["preview_url"],
            "/topik/preview/60/13/13",
        )
        self.assertEqual(
            result["topik_occurrences"][0]["preview_urls"],
            ["/topik/preview/60/13/13"],
        )

    def test_shared_preview_urls_expose_every_crop_index(self):
        result = {
            "topik_occurrences": [
                {"exam_number": 64, "question_start": 49, "question_end": 49}
            ]
        }

        with patch.object(
            topik_preview_service,
            "metadata",
            {"64:49:49": {"crops": [{}, {}]}},
        ):
            add_topik_preview_urls(result)

        self.assertEqual(
            result["topik_occurrences"][0]["preview_urls"],
            [
                "/topik/preview/64/49/49",
                "/topik/preview/64/49/49?crop_index=1",
            ],
        )

    def test_individual_topik_preview_route_returns_png(self):
        png = b"\x89PNG\r\n\x1a\n"

        with patch(
            "app.main.KoreanGrammarRAG",
            return_value=SimpleNamespace(
                entries=[],
                topik=SimpleNamespace(units=[]),
            ),
        ), patch(
            "app.main.make_preview_response",
            return_value=Response(content=png, media_type="image/png"),
        ):
            with TestClient(app) as client:
                for path in (
                    "/topik/preview/64/5/5",
                    "/topik/preview/64/7/7",
                    "/topik/preview/64/11/11",
                    "/topik/preview/64/19/19",
                    "/topik/preview/60/13/13",
                ):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200, path)
                    self.assertEqual(response.headers["content-type"], "image/png")
                    self.assertTrue(response.content.startswith(png))

    def test_http_home_health_and_ask_contracts(self):
        class FakeRag:
            entries = [1, 2]
            topik = SimpleNamespace(units=[1])

            def ask(self, question, explain=True):
                return {
                    "question": question,
                    "explanation": "answer" if explain else None,
                    "entries": [],
                    "topik_occurrences": [],
                    "sources": [],
                }

        with patch("app.main.KoreanGrammarRAG", FakeRag):
            with TestClient(app) as client:
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(
                    client.get("/health").json(),
                    {
                        "status": "ok",
                        "dictionary_entries": 2,
                        "topik_documents": 1,
                    },
                )
                response = client.post(
                    "/ask", json={"question": "question", "explain": True}
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["explanation"], "answer")
