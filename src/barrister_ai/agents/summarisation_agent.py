"""Case Summarisation Agent.

Receives a user query whose intent is ``case_summarisation`` and returns a
structured summary by retrieving matching precedents via TF-IDF search and
optionally enriching the answer through the configured LLM.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from barrister_ai.config import get_settings
from barrister_ai.retrieval.tfidf_search import LegalPrecedentSearchEngine

logger = logging.getLogger(__name__)

_ENGINE_LOCK = threading.Lock()
_ENGINE: LegalPrecedentSearchEngine | None = None


def _get_engine() -> LegalPrecedentSearchEngine | None:
    """Lazily initialise and cache the TF-IDF precedent search engine."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            dataset_path = (
                get_settings().knowledge_base_dir / "generated" / "legal_cases.json"
            )
            if not dataset_path.exists():
                logger.warning(
                    "SummarisationAgent: dataset not found at %s. "
                    "Proceeding without TF-IDF results.",
                    dataset_path,
                )
                return None
            engine = LegalPrecedentSearchEngine(dataset_path=dataset_path)
            engine.initialize()
            _ENGINE = engine
    return _ENGINE


def _format_case(rank: int, item: dict[str, Any]) -> str:
    """Format a single retrieved case into a readable summary block."""
    case = item.get("case", {})
    lines: list[str] = [
        f"[{rank}] {case.get('case_id', 'N/A')} — {case.get('domain', 'General')}",
    ]
    for field, label in (
        ("court", "Court"),
        ("act", "Act"),
        ("section", "Section"),
        ("facts", "Facts"),
        ("ratio", "Ratio Decidendi"),
        ("decision", "Decision"),
        ("summary", "Summary"),
    ):
        value = case.get(field, "")
        if value:
            lines.append(f"  {label}: {value}")
    score = item.get("similarity_percentage", 0)
    lines.append(f"  Relevance: {score}%")
    return "\n".join(lines)


class SummarisationAgent:
    """Agent responsible for case summarisation tasks.

    Parameters
    ----------
    top_k:
        Number of top matching cases to retrieve.
    minimum_score:
        Minimum TF-IDF cosine similarity score for a case to be included.
    """

    AGENT_NAME = "SummarisationAgent"
    SUPPORTED_INTENT = "case_summarisation"

    def __init__(self, top_k: int = 3, minimum_score: float = 0.0001) -> None:
        self.top_k = top_k
        self.minimum_score = minimum_score

    def run(self, query: str, intent_result: dict[str, Any]) -> dict[str, Any]:
        """Execute case summarisation for *query*.

        Parameters
        ----------
        query:
            Original user query.
        intent_result:
            Output dict from :func:`~barrister_ai.intent.classifier.classify_intent`.

        Returns
        -------
        dict with keys ``agent``, ``intent``, ``query``, ``answer``, ``cases``.
        """
        logger.info("%s: processing query=%r", self.AGENT_NAME, query)

        engine = _get_engine()
        cases: list[dict[str, Any]] = []
        if engine is not None:
            try:
                cases = engine.search(
                    query, top_k=self.top_k, minimum_score=self.minimum_score
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: search error – %s", self.AGENT_NAME, exc)

        if cases:
            summary_blocks = [
                _format_case(item["rank"], item) for item in cases
            ]
            answer = (
                f"Here are the most relevant case summaries for your query:\n\n"
                + "\n\n".join(summary_blocks)
            )
        else:
            answer = (
                "No matching cases were found in the knowledge base for your "
                "summarisation request. Please refine your query or upload "
                "relevant case documents."
            )

        return {
            "agent": self.AGENT_NAME,
            "intent": self.SUPPORTED_INTENT,
            "query": query,
            "answer": answer,
            "cases": cases,
        }
