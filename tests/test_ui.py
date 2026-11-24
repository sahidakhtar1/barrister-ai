"""Unit tests for barrister_ai.ui."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from barrister_ai.ui import (
    TEMPLATES_DIR,
    create_ui_server,
    get_home_html_path,
    load_home_html,
)
from barrister_ai.ui.app import _UIRequestHandler


class _FakeSocket:
    def __init__(self, request_bytes: bytes) -> None:
        self._rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode: str, *_args, **_kwargs):
        return self._rfile if "r" in mode else self.wfile

    def sendall(self, data: bytes) -> None:
        self.wfile.write(data)


def _dispatch_get(path: str) -> bytes:
    sock = _FakeSocket(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode("ascii"))
    _UIRequestHandler(
        sock,
        ("127.0.0.1", 0),
        None,
        directory=str(TEMPLATES_DIR),
    )
    return sock.wfile.getvalue()


class TestUIModule(unittest.TestCase):
    def test_home_template_exists_and_loads(self) -> None:
        path = get_home_html_path()
        self.assertTrue(path.exists())
        self.assertEqual(path.parent, TEMPLATES_DIR)
        html = load_home_html()
        self.assertIn("<form", html)
        self.assertIn("queryForm", html)

    def test_ui_handler_serves_health_and_index(self) -> None:
        health_raw = _dispatch_get("/health").decode("utf-8")
        self.assertIn("200 OK", health_raw)
        self.assertIn('{"status":"ok"}', health_raw)

        index_raw = _dispatch_get("/").decode("utf-8")
        self.assertIn("200 OK", index_raw)
        self.assertIn("PROCESS QUERY", index_raw)

    def test_create_ui_server_wires_templates_directory(self) -> None:
        with patch("barrister_ai.ui.app.ThreadingHTTPServer") as mock_server:
            create_ui_server(host="127.0.0.1", port=8080)
            mock_server.assert_called_once()
            addr, handler = mock_server.call_args[0]
            self.assertEqual(addr, ("127.0.0.1", 8080))
            self.assertEqual(handler.keywords["directory"], str(TEMPLATES_DIR))


if __name__ == "__main__":
    unittest.main()
