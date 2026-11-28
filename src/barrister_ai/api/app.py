"""FastAPI REST API for Barrister_AI legal query processing.

Query pipeline
--------------
1. Receive ``POST /barristerai/query`` with ``{"queryText": "..."}``
2. Call :func:`~barrister_ai.intent.router.route_query` which:
   a. Runs classical ML intent classification (TF-IDF + Logistic Regression)
   b. Dispatches to the matched specialised agent:
      - ``case_summarisation``  → :class:`~barrister_ai.agents.SummarisationAgent`
      - ``citation_validation`` → :class:`~barrister_ai.agents.CitationValidationAgent`
      - ``precedent_search``    → :class:`~barrister_ai.agents.PrecedentSearchAgent`
3. Return the structured agent response to the caller.

The legacy ``/baristerai/query`` route (typo) is kept for backwards compat.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from barrister_ai.config import VERSION
from barrister_ai.intent.router import route_query

QueryProcessor = Callable[[str], dict[str, Any]]


class QueryRequest(BaseModel):
    """Request payload for the legal query endpoint."""

    queryText: str


def create_api_app(query_processor: QueryProcessor | None = None) -> FastAPI:
    """Create and configure the FastAPI REST API application.

    Parameters
    ----------
    query_processor:
        Optional override callable ``(query: str) -> dict``.  Defaults to
        :func:`~barrister_ai.intent.router.route_query`.
    """
    processor = query_processor or route_query
    application = FastAPI(title="Barrister_AI API", version=VERSION)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/baristerai/query")
    @application.post("/barristerai/query", include_in_schema=False)
    def query_endpoint(payload: QueryRequest) -> dict[str, Any]:
        """Accept a natural-language legal query, classify its intent and
        route it to the appropriate specialised agent.

        Returns the agent's structured response including:
        - ``intent``      – detected intent label
        - ``confidence``  – classifier confidence score
        - ``agent``       – name of the agent that handled the request
        - ``answer``      – human-readable response
        - additional agent-specific fields (``cases``, ``validation``, ``precedents``)
        """
        print("/barristerai/query")
        print(f"Query test = {payload.queryText}")
        cleaned = payload.queryText.strip()

        if not cleaned:
            raise HTTPException(
                status_code=400,
                detail="Field 'queryText' is required and must be a non-empty string.",
            )
        try:
            return processor(cleaned)
        except Exception as exc:
            print("Exeption in /barristerai/query")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process query: {exc}",
            ) from exc

    return application


app = create_api_app()


def serve_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Barrister_AI REST API server with Uvicorn."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve_api()
