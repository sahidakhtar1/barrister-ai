"""Intent-based query router for Barrister_AI.

Flow
----
User query
  └─► :func:`classify_intent`   (classical ML – TF-IDF + LR)
        └─► intent label
              └─► dispatch to the matching specialised agent
                    ├─ case_summarisation  → :class:`SummarisationAgent`
                    ├─ citation_validation → :class:`CitationValidationAgent`
                    └─ precedent_search   → :class:`PrecedentSearchAgent`
"""

from __future__ import annotations

import logging
from typing import Any

from barrister_ai.intent.classifier import (
    classify_intent,
    INTENT_CASE_SUMMARISATION,
    INTENT_CITATION_VALIDATION,
    INTENT_PRECEDENT_SEARCH,
)
from barrister_ai.agents.summarisation_agent import SummarisationAgent
from barrister_ai.agents.citation_agent import CitationValidationAgent
from barrister_ai.agents.precedent_agent import PrecedentSearchAgent

logger = logging.getLogger(__name__)


class IntentRouter:
    """Routes an incoming query to the correct specialised agent.

    Parameters
    ----------
    summarisation_agent:
        Agent instance for ``case_summarisation`` intents.
        Defaults to a new :class:`~barrister_ai.agents.SummarisationAgent`.
    citation_agent:
        Agent instance for ``citation_validation`` intents.
        Defaults to a new :class:`~barrister_ai.agents.CitationValidationAgent`.
    precedent_agent:
        Agent instance for ``precedent_search`` intents.
        Defaults to a new :class:`~barrister_ai.agents.PrecedentSearchAgent`.
    intent_classifier:
        Callable ``(query: str) -> dict`` used to determine intent.
        Defaults to the module-level :func:`~barrister_ai.intent.classifier.classify_intent`.
    """

    def __init__(
        self,
        summarisation_agent: SummarisationAgent | None = None,
        citation_agent: CitationValidationAgent | None = None,
        precedent_agent: PrecedentSearchAgent | None = None,
        intent_classifier: Any | None = None,
    ) -> None:
        self._summarisation = summarisation_agent or SummarisationAgent()
        self._citation = citation_agent or CitationValidationAgent()
        self._precedent = precedent_agent or PrecedentSearchAgent()
        self._classify = intent_classifier or classify_intent

        self._dispatch: dict[str, Any] = {
            INTENT_CASE_SUMMARISATION: self._summarisation,
            INTENT_CITATION_VALIDATION: self._citation,
            INTENT_PRECEDENT_SEARCH: self._precedent,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str) -> dict[str, Any]:
        """Classify *query* then dispatch to the appropriate agent.

        Parameters
        ----------
        query:
            Raw natural-language legal query from the user.

        Returns
        -------
        dict with at least the following keys:

        * ``query``         – original query string
        * ``intent``        – classified intent label
        * ``confidence``    – classifier confidence (0–1)
        * ``above_threshold`` – whether confidence met the threshold
        * ``agent``         – name of the agent that handled the query
        * ``answer``        – human-readable response
        * All additional keys returned by the chosen agent (e.g. ``cases``,
          ``validation``, ``precedents``)
        """
        print(f"Route for {query}")
        cleaned = (query or "").strip()
        if not cleaned:
            raise ValueError("query must be a non-empty string.")

        # --- Step 1: classify intent via classical ML ---
        intent_result = self._classify(cleaned)
        intent = intent_result["intent"]

        logger.info(
            "IntentRouter: query=%r → intent=%r (confidence=%.3f, above_threshold=%s)",
            cleaned,
            intent,
            intent_result.get("confidence", 0.0),
            intent_result.get("above_threshold"),
        )

        # --- Step 2: dispatch to the matching agent ---
        agent = self._dispatch.get(intent)
        if agent is None:
            # Safety net – should never happen with the default classifier
            logger.error(
                "IntentRouter: no agent registered for intent %r; "
                "falling back to PrecedentSearchAgent.",
                intent,
            )
            agent = self._precedent

        agent_result = agent.run(cleaned, intent_result)

        # --- Step 3: merge classifier metadata into the final response ---
        return {
            **intent_result,          # query, intent, confidence, all_scores, above_threshold
            **agent_result,           # agent, answer, + agent-specific keys
        }


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_DEFAULT_ROUTER: IntentRouter | None = None


def _get_default_router() -> IntentRouter:
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = IntentRouter()
    return _DEFAULT_ROUTER


def route_query(query: str) -> dict[str, Any]:
    """Classify *query* and route it to the appropriate agent.

    This is the primary callable expected by the API layer.

    Parameters
    ----------
    query:
        Raw natural-language legal query from the user.

    Returns
    -------
    dict – combined classifier + agent response.
    """
    return _get_default_router().route(query)
