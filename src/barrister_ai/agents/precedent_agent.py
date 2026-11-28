"""Precedent Search Agent.

Receives a user query whose intent is ``precedent_search`` and returns
the most semantically similar case law from the TF-IDF index, enriched
with structured metadata.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from barrister_ai.config import get_settings
from barrister_ai.citation_validation.parser import extract_citation_details
from barrister_ai.retrieval.tfidf_search import LegalPrecedentSearchEngine

logger = logging.getLogger(__name__)

_ENGINE_LOCK = threading.Lock()
_ENGINE: LegalPrecedentSearchEngine | None = None


def _get_engine() -> LegalPrecedentSearchEngine | None:
    """Lazily initialise and cache the shared TF-IDF search engine."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            dataset_path = (
                get_settings().knowledge_base_dir / "generated" / "legal_cases.json"
            )
            if not dataset_path.exists():
                logger.warning(
                    "PrecedentSearchAgent: dataset not found at %s.",
                    dataset_path,
                )
                return None
            engine = LegalPrecedentSearchEngine(dataset_path=dataset_path)
            engine.initialize()
            _ENGINE = engine
    return _ENGINE


def _format_precedent(item: dict[str, Any]) -> dict[str, Any]:
    """Return a slim, serialisable precedent dict from a raw search result."""
    case = item.get("case", {})
    return {
        "rank": item.get("rank"),
        "similarity_percentage": item.get("similarity_percentage"),
        "case_id": case.get("case_id", "N/A"),
        "domain": case.get("domain", ""),
        "court": case.get("court", ""),
        "act": case.get("act", ""),
        "section": case.get("section", ""),
        "facts": case.get("facts", ""),
        "ratio": case.get("ratio", ""),
        "decision": case.get("decision", ""),
    }


class PrecedentSearchAgent:
    """Agent responsible for precedent / case-law search.

    Parameters
    ----------
    top_k:
        Maximum number of precedents to retrieve.
    minimum_score:
        Minimum cosine similarity required for a result to be included.
    """

    AGENT_NAME = "PrecedentSearchAgent"
    SUPPORTED_INTENT = "precedent_search"

    def __init__(self, top_k: int = 5, minimum_score: float = 0.0001) -> None:
        self.top_k = top_k
        self.minimum_score = minimum_score

    def run(self, query: str, intent_result: dict[str, Any]) -> dict[str, Any]:
        """Execute precedent search for *query*.

        Additionally tries to extract any embedded citation reference from the
        query and includes it in the response metadata.

        Parameters
        ----------
        query:
            Original user query.
        intent_result:
            Output dict from :func:`~barrister_ai.intent.classifier.classify_intent`.

        Returns
        -------
        dict with keys ``agent``, ``intent``, ``query``, ``answer``,
        ``precedents``, ``citation_hint``.
        """
        logger.info("%s: processing query=%r", self.AGENT_NAME, query)

        # Try to extract any inline citation hint from the query
        parsed_citation = extract_citation_details(query)
        citation_hint = None if "Error" in parsed_citation else parsed_citation

        engine = _get_engine()
        raw_results: list[dict[str, Any]] = []
        if engine is not None:
            try:
                raw_results = engine.search(
                    query, top_k=self.top_k, minimum_score=self.minimum_score
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: search error – %s", self.AGENT_NAME, exc)

        precedents = [_format_precedent(r) for r in raw_results]

        if precedents:
            lines = ["Relevant legal precedents:\n"]
            for p in precedents:
                line = (
                    f"[{p['rank']}] {p['case_id']} | {p['domain']} "
                    f"| {p['court']} ({p['similarity_percentage']}% match)"
                )
                if p.get("facts"):
                    line += f"\n   Facts   : {p['facts']}"
                if p.get("ratio"):
                    line += f"\n   Ratio   : {p['ratio']}"
                if p.get("decision"):
                    line += f"\n   Decision: {p['decision']}"
                lines.append(line)
            answer = "\n\n".join(lines)
        else:
            answer = (
                "No matching precedents were found for your query. "
                "Consider broadening your search terms or uploading "
                "domain-specific case documents."
            )

        return {
            "agent": self.AGENT_NAME,
            "intent": self.SUPPORTED_INTENT,
            "query": query,
            "answer": answer,
            "precedents": precedents,
            "citation_hint": citation_hint,
        }
