"""Tests for the IntentClassifier (classical ML)."""

from __future__ import annotations

import pytest

from barrister_ai.intent.classifier import (
    IntentClassifier,
    classify_intent,
    INTENT_CASE_SUMMARISATION,
    INTENT_CITATION_VALIDATION,
    INTENT_PRECEDENT_SEARCH,
    ALL_INTENTS,
)


@pytest.fixture(scope="module")
def clf() -> IntentClassifier:
    return IntentClassifier()


class TestIntentClassifierBasic:
    def test_summarisation_intent(self, clf: IntentClassifier) -> None:
        result = clf.classify("summarise the case of Maneka Gandhi vs Union of India")
        assert result["intent"] == INTENT_CASE_SUMMARISATION

    def test_citation_validation_intent(self, clf: IntentClassifier) -> None:
        result = clf.classify("validate Section 302 of the Indian Penal Code 1860")
        assert result["intent"] == INTENT_CITATION_VALIDATION

    def test_precedent_search_intent(self, clf: IntentClassifier) -> None:
        result = clf.classify("find precedents on right to privacy under Article 21")
        assert result["intent"] == INTENT_PRECEDENT_SEARCH

    def test_result_has_required_keys(self, clf: IntentClassifier) -> None:
        result = clf.classify("summarize judgment in Vishakha case")
        assert set(result.keys()) >= {"query", "intent", "confidence", "all_scores", "above_threshold"}

    def test_all_scores_contains_all_intents(self, clf: IntentClassifier) -> None:
        result = clf.classify("find case law on motor accident")
        for intent in ALL_INTENTS:
            assert intent in result["all_scores"]

    def test_confidence_range(self, clf: IntentClassifier) -> None:
        result = clf.classify("check whether Section 420 IPC is valid")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_query_raises(self, clf: IntentClassifier) -> None:
        with pytest.raises(ValueError):
            clf.classify("")

    def test_whitespace_only_raises(self, clf: IntentClassifier) -> None:
        with pytest.raises(ValueError):
            clf.classify("   ")


class TestIntentClassifierFitExtra:
    def test_fit_extra_adds_examples(self) -> None:
        clf = IntentClassifier()
        clf.fit_extra(
            ["summarise the Puttaswamy privacy judgment briefly"],
            [INTENT_CASE_SUMMARISATION],
        )
        result = clf.classify("summarise the Puttaswamy privacy judgment briefly")
        assert result["intent"] == INTENT_CASE_SUMMARISATION

    def test_fit_extra_mismatched_lengths_raises(self) -> None:
        clf = IntentClassifier()
        with pytest.raises(ValueError):
            clf.fit_extra(["query one", "query two"], [INTENT_PRECEDENT_SEARCH])

    def test_fit_extra_unknown_label_raises(self) -> None:
        clf = IntentClassifier()
        with pytest.raises(ValueError):
            clf.fit_extra(["some query"], ["unknown_intent"])


class TestClassifyIntentConvenienceFunction:
    def test_classify_intent_returns_dict(self) -> None:
        result = classify_intent("find precedents for cheque bounce Section 138")
        assert isinstance(result, dict)
        assert "intent" in result

    def test_classify_intent_summarisation(self) -> None:
        result = classify_intent("give me a brief summary of the Kesavananda Bharati case")
        assert result["intent"] == INTENT_CASE_SUMMARISATION
