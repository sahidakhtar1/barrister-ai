"""Project structure validation script for Barrister_AI."""

from __future__ import annotations

from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import get_settings  # noqa: E402

REQUIRED_FOLDERS = [
    "data/knowledge_base",
    "data/knowledge_base/acts",
    "data/knowledge_base/courts",
    "data/knowledge_base/domains",
    "data/knowledge_base/templates",
    "data/knowledge_base/generated",
    "data/acts",
    "data/precedents",
    "src/barrister_ai",
    "src/barrister_ai/citation_validation",
    "src/barrister_ai/retrieval",
    "src/barrister_ai/classification",
    "src/barrister_ai/ui",
    "docs",
    "models",
    "tests",
    "reports",
    "presentation",
]

REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
    "src/barrister_ai/config.py",
    "src/barrister_ai/main.py",
    "src/barrister_ai/ui/templates/index.html",
]


def main() -> int:
    """Validate project folder and core file layout."""
    settings = get_settings()
    passed = 0
    failed = 0

    print("=" * 60)
    print("BarristerAI Validation Report")
    print("=" * 60)
    print(f"\nPython Version : {platform.python_version()}")
    print("\nChecking Folders")

    for folder in REQUIRED_FOLDERS:
        if (settings.root_dir / folder).exists():
            print(f"PASS   {folder}")
            passed += 1
        else:
            print(f"FAIL   {folder}")
            failed += 1

    print("\nChecking Files")
    for file in REQUIRED_FILES:
        if (settings.root_dir / file).exists():
            print(f"PASS   {file}")
            passed += 1
        else:
            print(f"FAIL   {file}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Passed : {passed}")
    print(f"Failed : {failed}")
    status = "READY" if failed == 0 else "FIX ISSUES"
    print(f"\nPROJECT STATUS : {status}")
    print("=" * 60)

    report = settings.reports_dir / "M1_Validation_Report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "BarristerAI Validation\n"
        + "=" * 40
        + f"\nPassed : {passed}\nFailed : {failed}\n\nSTATUS : {status}\n",
        encoding="utf-8",
    )
    print(f"\nValidation report generated: {report}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
