"""Tests for the IntentRouter and agent dispatch."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from barrister_ai.intent.router import IntentRouter, route_query
from barrister_ai.intent.classifier import (
    INTENT_CASE_SUMMARISATION,
    INTENT_CITATION_VALIDATION,
    INTENT_PRECEDENT_SEARCH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent(agent_name: str, intent: str) -> MagicMock:
    agent = MagicMock()
    agent.run.return_value = {
        "agent": agent_name,
        "intent": intent,
        "query": "mock_query",
        "answer": f"Mock answer from {agent_name}",
    }
    return agent


def _make_mock_classifier(intent: str, confidence: float = 0.9) -> MagicMock:
    def _classify(query: str) -> dict[str, Any]:
        return {
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "all_scores": {intent: confidence},
            "above_threshold": confidence >= 0.4,
        }
    return _classify


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIntentRouter:
    def _router_for_intent(self, intent: str) -> tuple[IntentRouter, MagicMock, MagicMock, MagicMock]:
        summarisation = _make_mock_agent("SummarisationAgent", INTENT_CASE_SUMMARISATION)
        citation = _make_mock_agent("CitationValidationAgent", INTENT_CITATION_VALIDATION)
        precedent = _make_mock_agent("PrecedentSearchAgent", INTENT_PRECEDENT_SEARCH)
        classifier = _make_mock_classifier(intent)
        router = IntentRouter(
            summarisation_agent=summarisation,
            citation_agent=citation,
            precedent_agent=precedent,
            intent_classifier=classifier,
        )
        return router, summarisation, citation, precedent

    def test_routes_to_summarisation_agent(self) -> None:
        router, summarisation, citation, precedent = self._router_for_intent(
            INTENT_CASE_SUMMARISATION
        )
        result = router.route("summarise Maneka Gandhi case")
        summarisation.run.assert_called_once()
        citation.run.assert_not_called()
        precedent.run.assert_not_called()
        assert result["agent"] == "SummarisationAgent"

    def test_routes_to_citation_agent(self) -> None:
        router, summarisation, citation, precedent = self._router_for_intent(
            INTENT_CITATION_VALIDATION
        )
        result = router.route("validate Section 302 IPC")
        citation.run.assert_called_once()
        summarisation.run.assert_not_called()
        precedent.run.assert_not_called()
        assert result["agent"] == "CitationValidationAgent"

    def test_routes_to_precedent_agent(self) -> None:
        router, summarisation, citation, precedent = self._router_for_intent(
            INTENT_PRECEDENT_SEARCH
        )
        result = router.route("find case law on privacy rights")
        precedent.run.assert_called_once()
        summarisation.run.assert_not_called()
        citation.run.assert_not_called()
        assert result["agent"] == "PrecedentSearchAgent"

    def test_result_includes_classifier_metadata(self) -> None:
        router, *_ = self._router_for_intent(INTENT_PRECEDENT_SEARCH)
        result = router.route("precedents on anticipatory bail")
        assert "intent" in result
        assert "confidence" in result
        assert "above_threshold" in result

    def test_empty_query_raises(self) -> None:
        router, *_ = self._router_for_intent(INTENT_PRECEDENT_SEARCH)
        with pytest.raises(ValueError):
            router.route("")

    def test_whitespace_only_raises(self) -> None:
        router, *_ = self._router_for_intent(INTENT_PRECEDENT_SEARCH)
        with pytest.raises(ValueError):
            router.route("   ")


class TestRouteQueryConvenienceFunction:
    """Integration-style test – uses the real classifier (no DB/LLM calls)."""

    def test_returns_dict(self) -> None:
        # We use a known summarisation query so the real classifier fires
        result = route_query("summarise the case of Maneka Gandhi vs Union of India")
        assert isinstance(result, dict)
        assert "intent" in result
        assert "agent" in result
        assert "answer" in result

    def test_intent_is_valid(self) -> None:
        from barrister_ai.intent.classifier import ALL_INTENTS
        result = route_query("find precedents for cheque bounce Section 138 NI Act")
        assert result["intent"] in ALL_INTENTS
