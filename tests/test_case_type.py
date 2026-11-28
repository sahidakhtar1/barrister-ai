"""Unit tests for barrister_ai.classification.case_type."""

from __future__ import annotations

import unittest

import numpy as np

from barrister_ai.classification import CaseTypeClassifier


class _FakeEncoder:
    """Deterministic keyword-projection encoder for fast unit testing."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "flight" in lower or "product" in lower else 0.0,
                    1.0 if "theft" in lower or "murder" in lower else 0.0,
                    1.0 if "tax" in lower or "income" in lower else 0.0,
                ]
            )
        return np.array(vectors, dtype=float)


class TestCaseTypeClassifier(unittest.TestCase):
    def test_fit_predict_and_partial_fit(self) -> None:
        clf = CaseTypeClassifier(encoder=_FakeEncoder(), probability_threshold=0.4)
        queries = [
            "flight delayed compensation",
            "defective product refund",
            "murder weapon evidence",
            "theft case criminal",
            "income tax assessment",
            "tax penalty dispute",
        ]
        labels = ["Consumer", "Consumer", "Criminal", "Criminal", "Tax", "Tax"]

        clf.fit(queries, labels)
        pred = clf.predict("flight delayed refund")
        self.assertEqual(pred["predicted_case_type"], "Consumer")
        self.assertIn("Consumer", pred["probabilities"])

        clf.partial_fit(["new tax filing"], ["Tax"], all_classes=["Consumer", "Criminal", "Tax"])
        pred_after = clf.predict("income tax dispute")
        self.assertEqual(pred_after["predicted_case_type"], "Tax")


if __name__ == "__main__":
    unittest.main()
