"""Web UI module for serving the Barrister_AI query interface."""

from barrister_ai.ui.app import (
    TEMPLATES_DIR,
    create_ui_server,
    get_home_html_path,
    load_home_html,
    serve_ui,
)

__all__ = [
    "TEMPLATES_DIR",
    "create_ui_server",
    "get_home_html_path",
    "load_home_html",
    "serve_ui",
]
