"""M5.5 Step 2: Generate balanced positive and negative legal precedent pairs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import KNOWLEDGE_BASE, PRECEDENTS_DIR  # noqa: E402

INPUT_FILE = KNOWLEDGE_BASE / "generated" / "legal_cases_hardened.json"
OUTPUT_FILE = PRECEDENTS_DIR / "hardened_precedent_pairs.json"

RANDOM_SEED = 42
POSITIVES_PER_SOURCE = 4
HARD_NEGATIVES_PER_SOURCE = 2
CROSS_NEGATIVES_PER_SOURCE = 2


def load_cases(file_path: Path) -> list[dict[str, Any]]:
    """Load and validate hardened legal cases."""
    if not file_path.exists():
        raise FileNotFoundError(f"Hardened dataset not found: {file_path}")
    cases = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a non-empty JSON list.")
    return cases


def generate_pairs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate balanced positive, hard-negative, and cross-domain pairs."""
    rng = random.Random(RANDOM_SEED)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in cases:
        by_source[case["source_case_id"]].append(case)
        by_domain[case["domain"]].append(case)

    pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for source_id, source_cases in by_source.items():
        query = rng.choice(source_cases)
        positive_candidates = [c for c in source_cases if c["case_id"] != query["case_id"]]
        same_domain_candidates = [
            c for c in by_domain[query["domain"]] if c["source_case_id"] != source_id
        ]
        cross_domain_candidates = [c for c in cases if c["domain"] != query["domain"]]

        selections = [
            (
                rng.sample(positive_candidates, min(POSITIVES_PER_SOURCE, len(positive_candidates))),
                1,
                "positive_variant",
            ),
            (
                rng.sample(
                    same_domain_candidates,
                    min(HARD_NEGATIVES_PER_SOURCE, len(same_domain_candidates)),
                ),
                0,
                "hard_negative_same_domain",
            ),
            (
                rng.sample(
                    cross_domain_candidates,
                    min(CROSS_NEGATIVES_PER_SOURCE, len(cross_domain_candidates)),
                ),
                0,
                "negative_cross_domain",
            ),
        ]

        for candidates, label, pair_type in selections:
            for candidate in candidates:
                pair_key = (query["case_id"], candidate["case_id"])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pairs.append(
                    {
                        "query_case_id": query["case_id"],
                        "candidate_case_id": candidate["case_id"],
                        "query_source_id": query["source_case_id"],
                        "candidate_source_id": candidate["source_case_id"],
                        "query_domain": query["domain"],
                        "candidate_domain": candidate["domain"],
                        "label": label,
                        "pair_type": pair_type,
                    }
                )

    rng.shuffle(pairs)
    return pairs


def main() -> None:
    """Run M5.5 Step 2."""
    cases = load_cases(INPUT_FILE)
    pairs = generate_pairs(cases)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"M5.5 Step 2 complete: {len(pairs)} pairs -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
