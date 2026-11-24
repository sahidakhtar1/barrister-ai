"""Knowledge base generator for acts, courts, and legal domains."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import KNOWLEDGE_BASE  # noqa: E402

ACTS = [
    {"id": "ACT001", "name": "Negotiable Instruments Act", "year": 1881, "section": "138"},
    {"id": "ACT002", "name": "Bharatiya Nyaya Sanhita", "year": 2023, "section": "103"},
    {"id": "ACT003", "name": "Information Technology Act", "year": 2000, "section": "66"},
    {"id": "ACT004", "name": "Consumer Protection Act", "year": 2019, "section": "35"},
    {"id": "ACT005", "name": "Motor Vehicles Act", "year": 1988, "section": "166"},
]

COURTS = [
    {"id": "COURT001", "name": "Supreme Court of India"},
    {"id": "COURT002", "name": "Delhi High Court"},
    {"id": "COURT003", "name": "Bombay High Court"},
    {"id": "COURT004", "name": "Gujarat High Court"},
    {"id": "COURT005", "name": "District Court"},
]

DOMAINS = [
    {"id": "DOMAIN001", "name": "Cheque Bounce", "act_id": "ACT001"},
    {"id": "DOMAIN002", "name": "Murder", "act_id": "ACT002"},
    {"id": "DOMAIN003", "name": "Cyber Crime", "act_id": "ACT003"},
    {"id": "DOMAIN004", "name": "Consumer Complaint", "act_id": "ACT004"},
    {"id": "DOMAIN005", "name": "Motor Accident", "act_id": "ACT005"},
]


def main() -> None:
    """Write acts, courts, and domains JSON files into data/knowledge_base."""
    for sub, payload in [
        (KNOWLEDGE_BASE / "acts" / "acts.json", ACTS),
        (KNOWLEDGE_BASE / "courts" / "courts.json", COURTS),
        (KNOWLEDGE_BASE / "domains" / "domains.json", DOMAINS),
    ]:
        sub.parent.mkdir(parents=True, exist_ok=True)
        sub.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    print("Knowledge Base Generated Successfully")
    print(f"Acts: {len(ACTS)}, Courts: {len(COURTS)}, Domains: {len(DOMAINS)}")


if __name__ == "__main__":
    main()
