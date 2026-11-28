"""REST API package for Barrister_AI."""

from barrister_ai.api.app import (
    QueryRequest,
    app,
    create_api_app,
    process_query,
    serve_api,
)

__all__ = [
    "QueryRequest",
    "app",
    "create_api_app",
    "process_query",
    "serve_api",
]
