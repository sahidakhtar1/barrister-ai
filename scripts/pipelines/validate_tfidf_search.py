"""M4 pipeline: Validate the TF-IDF Legal Precedent Search Engine."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import KNOWLEDGE_BASE, REPORTS_DIR  # noqa: E402
from barrister_ai.retrieval import LegalPrecedentSearchEngine  # noqa: E402

DATASET_PATH = KNOWLEDGE_BASE / "generated" / "legal_cases.json"
REPORT_PATH = REPORTS_DIR / "m4_search_validation.json"

TEST_QUERIES = [
    {
        "name": "Cheque Bounce Search",
        "query": "Cheque issued by accused was dishonoured because of insufficient funds",
        "expected_terms": ["cheque", "dishonour", "insufficient"],
    },
    {
        "name": "Consumer Search",
        "query": "Consumer purchased a defective product and seller refused replacement or refund",
        "expected_terms": ["consumer", "defective", "refund"],
    },
    {
        "name": "Cyber Crime Search",
        "query": "Victim lost money through phishing email and fraudulent online bank transaction",
        "expected_terms": ["cyber", "phishing", "fraud"],
    },
    {
        "name": "Motor Accident Search",
        "query": "Road accident caused by negligent driving resulted in bodily injury and compensation claim",
        "expected_terms": ["accident", "negligent", "compensation"],
    },
    {
        "name": "Murder Search",
        "query": "Accused intentionally caused death using a deadly weapon",
        "expected_terms": ["murder", "death", "weapon"],
    },
]


def run_validation(engine: LegalPrecedentSearchEngine) -> list[dict[str, Any]]:
    """Run predefined legal search queries and collect validation records."""
    validation_results: list[dict[str, Any]] = []

    for test_number, test in enumerate(TEST_QUERIES, start=1):
        results = engine.search(query=test["query"], top_k=3, minimum_score=0.0)
        top_result = results[0] if results else None
        passed = bool(top_result and top_result["similarity_score"] > 0)

        validation_results.append(
            {
                "test_number": test_number,
                "test_name": test["name"],
                "query": test["query"],
                "expected_terms": test["expected_terms"],
                "results_returned": len(results),
                "top_similarity_score": top_result["similarity_score"] if top_result else 0.0,
                "top_similarity_percentage": (
                    top_result["similarity_percentage"] if top_result else 0.0
                ),
                "top_case": top_result["case"] if top_result else None,
                "passed": passed,
            }
        )

    return validation_results


def save_validation_report(
    statistics: dict[str, Any],
    validation_results: list[dict[str, Any]],
) -> None:
    """Save M4 validation output as JSON."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tests_passed = sum(1 for result in validation_results if result["passed"])

    report = {
        "module": "M4 - TF-IDF Legal Precedent Search",
        "status": "PASSED" if tests_passed == len(validation_results) else "FAILED",
        "statistics": statistics,
        "validation_summary": {
            "total_tests": len(validation_results),
            "tests_passed": tests_passed,
            "tests_failed": len(validation_results) - tests_passed,
        },
        "validation_results": validation_results,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    """Run TF-IDF search validation and save the report."""
    engine = LegalPrecedentSearchEngine(
        dataset_path=DATASET_PATH,
        max_features=5000,
        ngram_range=(1, 2),
    )
    engine.initialize()
    statistics = engine.get_statistics()
    validation_results = run_validation(engine)
    save_validation_report(statistics=statistics, validation_results=validation_results)

    tests_passed = sum(1 for result in validation_results if result["passed"])
    print(f"M4 Validation: {tests_passed}/{len(validation_results)} passed -> {REPORT_PATH}")
    return 0 if tests_passed == len(validation_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
