"""M5.5 Step 1: Create deterministic legal-language variants from synthetic cases."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import KNOWLEDGE_BASE  # noqa: E402

INPUT_FILE = KNOWLEDGE_BASE / "generated" / "legal_cases.json"
OUTPUT_FILE = KNOWLEDGE_BASE / "generated" / "legal_cases_hardened.json"

VARIANTS_PER_CASE = 5
RANDOM_SEED = 42

PROTECTED_FIELDS = {"case_id", "id", "case_number", "domain", "label", "outcome"}

LANGUAGE_VARIANTS = {
    "cheque dishonoured": [
        "cheque returned unpaid",
        "payment instrument was dishonoured",
        "bank declined the cheque",
    ],
    "insufficient funds": [
        "inadequate account balance",
        "insufficient balance",
        "lack of available funds",
    ],
    "consumer complaint": [
        "consumer grievance",
        "complaint before the consumer forum",
        "consumer dispute",
    ],
    "defective product": [
        "faulty goods",
        "product suffering from defects",
        "substandard merchandise",
    ],
    "online transaction": [
        "digital transaction",
        "internet-based payment",
        "electronic financial transaction",
    ],
    "cyber crime": [
        "digital offence",
        "computer-related offence",
        "online criminal activity",
    ],
    "motor accident": [
        "road traffic accident",
        "vehicular collision",
        "motor vehicle incident",
    ],
    "compensation": ["monetary relief", "financial compensation", "damages"],
    "murder": ["homicide", "fatal criminal act", "unlawful killing"],
    "accused": ["defendant", "person charged", "alleged offender"],
    "evidence": ["evidentiary material", "proof placed before the court", "supporting material"],
    "court": ["judicial forum", "trial court", "competent court"],
}

NEUTRAL_SENTENCES = [
    "The dispute required examination of the available record.",
    "The court assessed the facts and applicable legal principles.",
    "The parties relied on documentary and oral material.",
    "The matter involved interpretation of the relevant law.",
    "The decision depended on the circumstances of the case.",
]


def load_cases(file_path: Path) -> list[dict[str, Any]]:
    """Load and validate the original legal cases."""
    if not file_path.exists():
        raise FileNotFoundError(f"Original dataset not found: {file_path}")
    cases = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a non-empty JSON list of cases.")
    return cases


def replace_legal_language(text: str, rng: random.Random) -> tuple[str, bool]:
    """Replace legal expressions with alternative wording."""
    changed = False
    updated_text = text
    phrases = list(LANGUAGE_VARIANTS.items())
    rng.shuffle(phrases)

    for phrase, alternatives in phrases:
        if re.search(re.escape(phrase), updated_text, flags=re.IGNORECASE):
            replacement = rng.choice(alternatives)
            updated_text = re.sub(
                re.escape(phrase),
                replacement,
                updated_text,
                count=1,
                flags=re.IGNORECASE,
            )
            changed = True

    return updated_text, changed


def transform_value(value: Any, rng: random.Random) -> tuple[Any, bool]:
    """Recursively transform textual case content."""
    if isinstance(value, str):
        return replace_legal_language(value, rng)
    if isinstance(value, list):
        items, changed = [], False
        for item in value:
            transformed, item_changed = transform_value(item, rng)
            items.append(transformed)
            changed = changed or item_changed
        return items, changed
    if isinstance(value, dict):
        result, changed = {}, False
        for key, item in value.items():
            transformed, item_changed = transform_value(item, rng)
            result[key] = transformed
            changed = changed or item_changed
        return result, changed
    return value, False


def generate_hardened_dataset(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create originals plus language-varied cases."""
    rng = random.Random(RANDOM_SEED)
    hardened: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        original_id = str(
            case.get("case_id") or case.get("id") or case.get("case_number") or f"CASE{index + 1:04d}"
        )
        original = copy.deepcopy(case)
        original["case_id"] = original_id
        original["source_case_id"] = original_id
        original["language_variant"] = 0
        hardened.append(original)

        for variant_number in range(1, VARIANTS_PER_CASE + 1):
            variant = copy.deepcopy(case)
            changed = False
            for field, value in list(variant.items()):
                if field in PROTECTED_FIELDS:
                    continue
                transformed, field_changed = transform_value(value, rng)
                variant[field] = transformed
                changed = changed or field_changed

            if not changed:
                for field, value in variant.items():
                    if field not in PROTECTED_FIELDS and isinstance(value, str):
                        variant[field] = f"{value} {rng.choice(NEUTRAL_SENTENCES)}"
                        break

            variant["case_id"] = f"{original_id}_V{variant_number}"
            variant["source_case_id"] = original_id
            variant["language_variant"] = variant_number
            hardened.append(variant)

    return hardened


def main() -> None:
    """Run M5.5 Step 1."""
    cases = load_cases(INPUT_FILE)
    hardened_cases = generate_hardened_dataset(cases)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(hardened_cases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"M5.5 Step 1 complete: {len(hardened_cases)} cases -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
