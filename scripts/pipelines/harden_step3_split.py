"""M5.5 Step 3: Attach legal text and create leakage-safe train/test splits."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import KNOWLEDGE_BASE, PRECEDENTS_DIR  # noqa: E402

CASES_FILE = KNOWLEDGE_BASE / "generated" / "legal_cases_hardened.json"
PAIRS_FILE = PRECEDENTS_DIR / "hardened_precedent_pairs.json"
TRAIN_FILE = PRECEDENTS_DIR / "hardened_precedent_train.json"
TEST_FILE = PRECEDENTS_DIR / "hardened_precedent_test.json"

TEST_SIZE = 0.20
RANDOM_SEED = 42

TEXT_FIELDS = (
    "case_title",
    "title",
    "facts",
    "summary",
    "legal_issue",
    "issue",
    "legal_sections",
    "sections",
    "judgment",
    "decision",
    "reasoning",
)


def normalise_value(value: Any) -> str:
    """Convert supported values into clean text."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return " ".join(
            f"{key} {item}" for key, item in value.items() if str(item).strip()
        )
    return str(value).strip()


def build_case_text(case: dict[str, Any]) -> str:
    """Combine legal fields into one text document."""
    parts = [
        f"{field.replace('_', ' ')}: {normalise_value(case.get(field))}"
        for field in TEXT_FIELDS
        if normalise_value(case.get(field))
    ]
    return " ".join(parts).strip()


def main() -> None:
    """Run M5.5 Step 3."""
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    pairs = json.loads(PAIRS_FILE.read_text(encoding="utf-8"))
    case_lookup = {str(c["case_id"]): c for c in cases}

    records: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        query_id = pair["query_case_id"]
        candidate_id = pair["candidate_case_id"]
        records.append(
            {
                "pair_id": f"HPAIR{index:05d}",
                "query_case_id": query_id,
                "candidate_case_id": candidate_id,
                "query_source_id": pair["query_source_id"],
                "candidate_source_id": pair["candidate_source_id"],
                "query_text": build_case_text(case_lookup[query_id]),
                "candidate_text": build_case_text(case_lookup[candidate_id]),
                "label": int(pair["label"]),
                "pair_type": pair["pair_type"],
            }
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(
        splitter.split(
            X=records,
            y=[r["label"] for r in records],
            groups=[r["query_source_id"] for r in records],
        )
    )

    train_data = [records[i] for i in train_idx]
    test_data = [records[i] for i in test_idx]

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_FILE.write_text(json.dumps(train_data, indent=2, ensure_ascii=False), encoding="utf-8")
    TEST_FILE.write_text(json.dumps(test_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"M5.5 Step 3 complete: train={len(train_data)}, test={len(test_data)}")


if __name__ == "__main__":
    main()
