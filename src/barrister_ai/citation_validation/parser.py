"""Regex-based legal citation deconstruction and parsing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

ACT_YEAR_PATTERN = re.compile(r"\b([A-Z][a-zA-Z\s,\-\(\)]+?\s+Act),\s*(\d{4})\b")
SECTION_PATTERN = re.compile(
    r"\bSection\s+(\d+)(?:\s*(\([^)]+\)(?:\s*\([^)]+\))*))?",
    re.IGNORECASE,
)
CONNECTIVE_WORDS_PATTERN = re.compile(
    r"\b(under|of\s+the|in|by|the|according\s+to)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedCitation:
    """Structured representation of a parsed legal citation."""

    act_name: str
    year: str
    section: str | None = None
    sub_section: str | None = None
    proposition: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to the legacy dictionary schema."""
        return {
            "Section": self.section,
            "Sub-section": self.sub_section,
            "Act_Name": self.act_name,
            "Year": self.year,
            "Proposition": self.proposition,
        }


def parse_citation(text: str) -> ParsedCitation:
    """Parse a legal citation string into a :class:`ParsedCitation`.

    Raises:
        ValueError: If input is empty or missing an Act Name and Year.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Citation text must be a non-empty string.")

    clean_text = re.sub(r"\s+", " ", text).strip()
    act_match = ACT_YEAR_PATTERN.search(clean_text)

    if not act_match:
        raise ValueError(
            "Could not parse the required Act Name and Year in the given text."
        )

    act_name = act_match.group(1).strip()
    year = act_match.group(2)

    sec_match = SECTION_PATTERN.search(clean_text)
    section: str | None = None
    sub_section: str | None = None

    if sec_match:
        section = sec_match.group(1)
        if sec_match.group(2):
            sub_section = sec_match.group(2).strip()

    proposition = clean_text.replace(act_match.group(0), "")
    if sec_match:
        proposition = proposition.replace(sec_match.group(0), "")

    proposition = CONNECTIVE_WORDS_PATTERN.sub(" ", proposition)
    proposition = re.sub(r"\s+", " ", proposition).strip(" .,")

    return ParsedCitation(
        act_name=act_name,
        year=year,
        section=section,
        sub_section=sub_section,
        proposition=proposition,
    )


def extract_citation_details(text: str) -> dict[str, Any]:
    """Parse legal citations into a dictionary of components or an error dict.

    Extracts:
    - Section (optional)
    - Sub-section (optional)
    - Act_Name (required)
    - Year (required)
    - Proposition (remaining text after removing extracted citation patterns)
    """
    try:
        return parse_citation(text).to_dict()
    except ValueError as exc:
        return {
            "Error": str(exc),
            "Original_Text": text,
        }
