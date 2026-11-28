"""Web UI module for serving the Barrister_AI frontend interface."""

from barrister_ai.ui.app import (
    TEMPLATES_DIR,
    app,
    create_app,
    get_home_html_path,
    load_home_html,
    serve_ui,
)

__all__ = [
    "TEMPLATES_DIR",
    "app",
    "create_app",
    "get_home_html_path",
    "load_home_html",
    "serve_ui",
]
