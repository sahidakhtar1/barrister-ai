"""Unit tests for barrister_ai.ui."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from barrister_ai.ui import (
    TEMPLATES_DIR,
    app,
    create_app,
    get_home_html_path,
    load_home_html,
    serve_ui,
)


class TestUIModule(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_home_template_exists_and_loads(self) -> None:
        path = get_home_html_path()
        self.assertTrue(path.exists())
        self.assertEqual(path.parent, TEMPLATES_DIR)
        html = load_home_html()
        self.assertIn("<form", html)
        self.assertIn("queryForm", html)
        self.assertIn("/baristerai/query", html)
        self.assertIn("JSON.stringify({ queryText })", html)
        self.assertIn('<button id="submitBtn" type="submit" disabled>', html)
        self.assertIn("input.value.trim().length === 0", html)

    def test_ui_serves_health_and_index_only(self) -> None:
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertEqual(health_resp.json(), {"status": "ok"})

        index_resp = self.client.get("/")
        self.assertEqual(index_resp.status_code, 200)
        self.assertIn("text/html", index_resp.headers["content-type"])
        self.assertIn("PROCESS QUERY", index_resp.text)

        api_resp = self.client.post("/baristerai/query", json={"queryText": "test"})
        self.assertEqual(api_resp.status_code, 404)

    def test_create_app_injects_custom_api_base_url(self) -> None:
        custom_client = TestClient(create_app(api_base_url="http://localhost:9000/"))
        resp = custom_client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn('window.BARRISTER_API_URL = "http://localhost:9000";', resp.text)

    def test_serve_ui_starts_uvicorn(self) -> None:
        with patch("barrister_ai.ui.app.uvicorn.run") as mock_run:
            serve_ui(host="127.0.0.1", port=8080)
            mock_run.assert_called_once_with(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    unittest.main()
