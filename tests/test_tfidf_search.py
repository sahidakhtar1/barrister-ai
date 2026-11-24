"""Unit tests for barrister_ai.retrieval.tfidf_search."""

from __future__ import annotations

import unittest

from barrister_ai.config import KNOWLEDGE_BASE
from barrister_ai.retrieval import LegalPrecedentSearchEngine


class TestLegalPrecedentSearchEngine(unittest.TestCase):
    def test_search_returns_relevant_domain(self) -> None:
        dataset_path = KNOWLEDGE_BASE / "generated" / "legal_cases.json"
        engine = LegalPrecedentSearchEngine(dataset_path=dataset_path)
        engine.initialize()

        stats = engine.get_statistics()
        self.assertEqual(stats["cases_indexed"], 50)
        self.assertGreater(stats["vocabulary_size"], 0)

        results = engine.search(
            "Cheque dishonoured because of insufficient funds",
            top_k=3,
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["case"]["domain"], "Cheque Bounce")
        self.assertGreater(results[0]["similarity_score"], 0.0)

    def test_empty_query_raises_value_error(self) -> None:
        dataset_path = KNOWLEDGE_BASE / "generated" / "legal_cases.json"
        engine = LegalPrecedentSearchEngine(dataset_path=dataset_path)
        engine.initialize()

        with self.assertRaises(ValueError):
            engine.search("   ")


if __name__ == "__main__":
    unittest.main()
