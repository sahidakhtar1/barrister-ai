"""Lightweight HTTP server and template helpers for the Barrister_AI Web UI."""

from __future__ import annotations

from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def get_home_html_path() -> Path:
    """Return the path to the home page HTML template."""
    return TEMPLATES_DIR / "index.html"


def load_home_html() -> str:
    """Load and return the HTML content for the home UI."""
    home_path = get_home_html_path()
    if not home_path.exists():
        raise FileNotFoundError(f"UI template not found: {home_path}")
    return home_path.read_text(encoding="utf-8")


class _UIRequestHandler(SimpleHTTPRequestHandler):
    """Serves static templates from the UI templates directory."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            payload = b'{"status":"ok"}'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path in {"/", ""}:
            self.path = "/index.html"
        super().do_GET()


def create_ui_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """Create a configured HTTP server serving the UI templates directory."""
    handler = partial(_UIRequestHandler, directory=str(TEMPLATES_DIR))
    return ThreadingHTTPServer((host, port), handler)


def serve_ui(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the Barrister_AI UI server and block until interrupted."""
    server = create_ui_server(host=host, port=port)
    print(f"Serving Barrister_AI UI at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down UI server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve_ui()
