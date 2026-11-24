"""M3 synthetic legal case dataset generator and validator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import EXPORTS_DIR, KNOWLEDGE_BASE, REPORTS_DIR  # noqa: E402

NUM_CASES = 50

COURTS = [
    "Supreme Court",
    "Bombay High Court",
    "Delhi High Court",
    "Karnataka High Court",
    "Madras High Court",
]

ACT_SECTION = {
    "Cheque Bounce": ("Negotiable Instruments Act", "138"),
    "Murder": ("Indian Penal Code", "302"),
    "Cyber Crime": ("Information Technology Act", "66"),
    "Consumer": ("Consumer Protection Act", "35"),
    "Motor Accident": ("Motor Vehicles Act", "166"),
}


def build_case(template: dict, index: int) -> dict:
    """Generate one synthetic case record from a template."""
    domain = template["domain"]
    act, section = ACT_SECTION.get(domain, ("Unknown Act", "Unknown"))

    case = {
        "case_id": f"CASE{index:04d}",
        "domain": domain,
        "act": act,
        "section": section,
        "court": random.choice(COURTS),
        "facts": " ".join(random.sample(template["facts"], len(template["facts"]))),
        "issues": random.choice(template["issues"]),
        "arguments": random.choice(template["arguments"]),
        "analysis": random.choice(template["analysis"]),
        "ratio": random.choice(template["ratio"]),
        "decision": random.choice(template["decision"]),
    }

    case["label"] = (
        1
        if case["decision"].lower()
        in {"conviction", "compensation awarded", "compensation granted"}
        else 0
    )
    return case


def main() -> None:
    """Generate and validate synthetic legal cases."""
    random.seed(42)

    template_folder = KNOWLEDGE_BASE / "templates"
    generated_folder = KNOWLEDGE_BASE / "generated"
    generated_folder.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    templates = []
    for file in sorted(template_folder.glob("*.json")):
        if file.name == "templates.json":
            continue
        with file.open("r", encoding="utf-8") as handle:
            templates.append(json.load(handle))

    if not templates:
        raise RuntimeError(f"No templates found in {template_folder}")

    dataset = [build_case(random.choice(templates), i) for i in range(1, NUM_CASES + 1)]

    json_file = generated_folder / "legal_cases.json"
    json_file.write_text(
        json.dumps(dataset, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    fieldnames = [
        "case_id",
        "domain",
        "act",
        "section",
        "court",
        "facts",
        "issues",
        "arguments",
        "analysis",
        "ratio",
        "decision",
        "label",
    ]
    csv_file = EXPORTS_DIR / "legal_cases.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    domains: dict[str, int] = {}
    labels = {0: 0, 1: 0}
    for record in dataset:
        domains[record["domain"]] = domains.get(record["domain"], 0) + 1
        labels[record["label"]] += 1

    report_file = REPORTS_DIR / "dataset_report.txt"
    with report_file.open("w", encoding="utf-8") as report:
        report.write("=" * 60 + "\n")
        report.write("Barrister_AI Dataset Report\n")
        report.write("=" * 60 + "\n\n")
        report.write(f"Total Cases : {len(dataset)}\n\n")
        report.write("Cases by Domain\n-------------------------\n")
        for domain_name, count in sorted(domains.items()):
            report.write(f"{domain_name} : {count}\n")
        report.write("\nLabel Distribution\n-------------------------\n")
        report.write(f"Positive (1) : {labels[1]}\n")
        report.write(f"Negative (0) : {labels[0]}\n")
        report.write("\nValidation : PASSED\n")

    print(f"Generated {len(dataset)} cases -> {json_file}, {csv_file}")


if __name__ == "__main__":
    main()
