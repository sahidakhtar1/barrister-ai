"""Unit tests for barrister_ai.api."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from barrister_ai.api import (
    app,
    create_api_app,
    process_query,
    serve_api,
)


class TestAPIModule(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_api_app())

    def test_health_route(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_post_query_endpoint_accepts_query_text(self) -> None:
        resp = self.client.post(
            "/baristerai/query",
            json={"queryText": "cheque dishonoured insufficient funds"},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["queryText"], "cheque dishonoured insufficient funds")
        self.assertIn("Matching Legal Precedents:", payload["answer"])
        self.assertTrue(payload["results"])

    def test_post_query_endpoint_validates_query_text(self) -> None:
        empty_resp = self.client.post("/baristerai/query", json={"queryText": "   "})
        self.assertEqual(empty_resp.status_code, 400)
        self.assertIn("queryText", empty_resp.json()["detail"])

        missing_field_resp = self.client.post("/baristerai/query", json={})
        self.assertEqual(missing_field_resp.status_code, 422)

        not_found_resp = self.client.post("/unknown/endpoint", json={"queryText": "hello"})
        self.assertEqual(not_found_resp.status_code, 404)

    def test_cors_headers_allow_cross_origin_ui(self) -> None:
        resp = self.client.options(
            "/baristerai/query",
            headers={
                "Origin": "http://127.0.0.1:8080",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("access-control-allow-origin", resp.headers)

    def test_process_query_detects_citation_and_precedents(self) -> None:
        result = process_query("Section 138 of Negotiable Instruments Act, 1881 cheque dishonour")
        self.assertIsNotNone(result["citation"])
        self.assertEqual(result["citation"]["Section"], "138")
        self.assertIn("Detected Citation:", result["answer"])
        self.assertTrue(result["results"])

    def test_create_api_app_uses_injected_query_processor(self) -> None:
        custom_app = create_api_app(query_processor=lambda q: {"queryText": q, "answer": "stub"})
        custom_client = TestClient(custom_app)
        resp = custom_client.post("/baristerai/query", json={"queryText": "test query"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"queryText": "test query", "answer": "stub"})

    def test_serve_api_starts_uvicorn(self) -> None:
        with patch("barrister_ai.api.app.uvicorn.run") as mock_run:
            serve_api(host="127.0.0.1", port=8000)
            mock_run.assert_called_once_with(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    unittest.main()
