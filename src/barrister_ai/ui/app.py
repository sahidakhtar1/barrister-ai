"""FastAPI web UI application and template helpers for Barrister_AI."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from barrister_ai.config import VERSION

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def get_home_html_path() -> Path:
    """Return the path to the home page HTML template."""
    return TEMPLATES_DIR / "index.html"


def load_home_html(api_base_url: str | None = None) -> str:
    """Load and return the HTML content for the home UI."""
    home_path = get_home_html_path()
    if not home_path.exists():
        raise FileNotFoundError(f"UI template not found: {home_path}")
    html = home_path.read_text(encoding="utf-8")

    configured_url = api_base_url or os.getenv("BARRISTER_API_URL")
    if configured_url:
        config_tag = f"<script>window.BARRISTER_API_URL = {json.dumps(configured_url.rstrip('/'))};</script>\n</head>"
        html = html.replace("</head>", config_tag, 1)
    return html


def create_app(api_base_url: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application serving the Web UI."""
    application = FastAPI(title="Barrister_AI Web UI", version=VERSION)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/", response_class=HTMLResponse)
    @application.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
    def home() -> HTMLResponse:
        return HTMLResponse(content=load_home_html(api_base_url=api_base_url))

    return application


app = create_app()


def serve_ui(
    host: str = "127.0.0.1",
    port: int = 8080,
    api_base_url: str | None = None,
) -> None:
    """Start the Barrister_AI Web UI server with Uvicorn."""
    target_app = app if api_base_url is None else create_app(api_base_url=api_base_url)
    uvicorn.run(target_app, host=host, port=port)


if __name__ == "__main__":
    serve_ui()
