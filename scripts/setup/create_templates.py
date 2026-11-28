"""Creates and validates legal case templates in data/knowledge_base/templates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from barrister_ai.config import KNOWLEDGE_BASE  # noqa: E402

TEMPLATES = {
    "cheque_bounce": {
        "domain": "Cheque Bounce",
        "facts": [
            "The accused issued a cheque towards repayment of debt.",
            "The cheque was dishonoured due to insufficient funds.",
            "A statutory notice was served upon the accused.",
        ],
        "issues": [
            "Whether Section 138 NI Act is attracted.",
            "Whether legal notice requirements were fulfilled.",
        ],
        "arguments": [
            "Complainant proved legally enforceable debt.",
            "Accused denied liability.",
        ],
        "analysis": [
            "Court examined bank memo and statutory notice.",
            "Court verified limitation period.",
        ],
        "ratio": [
            "Dishonour of cheque constitutes offence under NI Act if statutory conditions are satisfied."
        ],
        "decision": ["Conviction", "Acquittal"],
    },
    "murder": {
        "domain": "Murder",
        "facts": [
            "Accused allegedly caused death using a deadly weapon.",
            "Eyewitness testimony was produced.",
            "Medical evidence supported prosecution.",
        ],
        "issues": ["Whether ingredients of Section 302 IPC are established."],
        "arguments": [
            "Prosecution relied upon eyewitnesses.",
            "Defence questioned credibility.",
        ],
        "analysis": ["Court evaluated motive and forensic evidence."],
        "ratio": ["Prosecution must establish guilt beyond reasonable doubt."],
        "decision": ["Conviction", "Acquittal"],
    },
    "cyber_crime": {
        "domain": "Cyber Crime",
        "facts": [
            "Unauthorized access to computer resources.",
            "Victim suffered financial loss.",
        ],
        "issues": ["Applicability of Information Technology Act."],
        "arguments": [
            "Digital evidence linked accused.",
            "Defence challenged authenticity.",
        ],
        "analysis": ["Court examined electronic evidence."],
        "ratio": ["Electronic evidence must satisfy admissibility requirements."],
        "decision": ["Conviction", "Acquittal"],
    },
    "consumer": {
        "domain": "Consumer",
        "facts": [
            "Consumer purchased defective goods.",
            "Seller failed to provide replacement.",
        ],
        "issues": ["Whether deficiency in service exists."],
        "arguments": [
            "Consumer sought compensation.",
            "Seller denied manufacturing defect.",
        ],
        "analysis": ["Commission reviewed invoices and expert reports."],
        "ratio": ["Consumer entitled to compensation for proven deficiency."],
        "decision": ["Compensation Awarded", "Complaint Dismissed"],
    },
    "motor_accident": {
        "domain": "Motor Accident",
        "facts": [
            "Road accident caused bodily injuries.",
            "Insurance policy was valid.",
        ],
        "issues": ["Quantum of compensation."],
        "arguments": [
            "Claimant relied upon medical records.",
            "Insurance company disputed negligence.",
        ],
        "analysis": ["Tribunal assessed disability percentage."],
        "ratio": ["Compensation should be just and reasonable."],
        "decision": ["Compensation Granted", "Claim Rejected"],
    },
}


def main() -> int:
    """Create template JSON files and validate required keys."""
    template_folder = KNOWLEDGE_BASE / "templates"
    template_folder.mkdir(parents=True, exist_ok=True)

    for name, data in TEMPLATES.items():
        file_path = template_folder / f"{name}.json"
        file_path.write_text(json.dumps(data, indent=4), encoding="utf-8")

    required = {"domain", "facts", "issues", "arguments", "analysis", "ratio", "decision"}
    valid = True
    for file_path in template_folder.glob("*.json"):
        if file_path.name == "templates.json":
            continue
        obj = json.loads(file_path.read_text(encoding="utf-8"))
        missing = required - set(obj.keys())
        if missing:
            print(f"{file_path.name} missing keys: {sorted(missing)}")
            valid = False

    print("Validation Passed" if valid else "Validation Failed")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
