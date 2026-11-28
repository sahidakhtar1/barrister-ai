"""Project directory bootstrap utility for Barrister_AI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import get_settings  # noqa: E402

REQUIRED_FOLDERS = [
    "docs",
    "data/knowledge_base/acts",
    "data/knowledge_base/courts",
    "data/knowledge_base/domains",
    "data/knowledge_base/templates",
    "data/knowledge_base/generated",
    "data/knowledge_base/exports",
    "data/acts/pdfs",
    "data/precedents",
    "data/exports",
    "src/barrister_ai/citation_validation/db",
    "src/barrister_ai/citation_validation/vector_store",
    "src/barrister_ai/citation_validation/ingestion",
    "src/barrister_ai/retrieval",
    "src/barrister_ai/classification",
    "src/barrister_ai/ui/templates",
    "scripts/setup",
    "scripts/pipelines",
    "models",
    "reports",
    "logs",
    "tests",
    "presentation",
    "notebooks",
]


def main() -> None:
    """Ensure all standard project folders exist and write a bootstrap report."""
    settings = get_settings()
    created_folders = 0

    for folder in REQUIRED_FOLDERS:
        path = settings.root_dir / folder
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created_folders += 1

    report = settings.reports_dir / "BOOTSTRAP_REPORT.txt"
    report.parent.mkdir(parents=True, exist_ok=True)

    with report.open("w", encoding="utf-8") as handle:
        handle.write("=" * 60 + "\n")
        handle.write("Barrister_AI Bootstrap Report\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Generated : {datetime.now()}\n\n")
        handle.write(f"Folders Created : {created_folders}\n")
        handle.write(f"\nProject Root\n{settings.root_dir}\n")

    print("=" * 60)
    print(" Barrister_AI Bootstrap Completed")
    print("=" * 60)
    print(f"Project Root    : {settings.root_dir}")
    print(f"Folders Created : {created_folders}")
    print(f"Report          : {report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
