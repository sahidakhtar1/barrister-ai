"""End-to-end citation validation orchestrator."""

from __future__ import annotations

from typing import Any, Callable

from barrister_ai.citation_validation.db.repository import get_sections_by_act_and_year
from barrister_ai.citation_validation.llm_client import validate_citation_locally
from barrister_ai.citation_validation.parser import extract_citation_details

SectionFetcher = Callable[..., dict[str, Any]]
LLMValidator = Callable[[str, str], str]


def _build_act_context(act_name: str, sections: list[dict[str, Any]]) -> str:
    """Format retrieved Act sections into an LLM-ready context string."""
    parts = [act_name]
    for sec in sections:
        sec_num = sec.get("section_number") or ""
        sec_title = sec.get("section_title") or ""
        sub_sec = sec.get("sub-section") or ""
        parts.append(
            f"Section number: {sec_num}, Section title: {sec_title}, sub-section - {sub_sec}"
        )
    return ", ".join(parts)


class CitationValidator:
    """Coordinates citation parsing, corpus lookup, and LLM verification."""

    def __init__(
        self,
        section_fetcher: SectionFetcher | None = None,
        llm_validator: LLMValidator | None = None,
    ) -> None:
        self.section_fetcher = section_fetcher or get_sections_by_act_and_year
        self.llm_validator = llm_validator or (
            lambda act_context, citation: validate_citation_locally(
                act_context=act_context,
                citation=citation,
            )
        )

    def validate(self, citation: str) -> dict[str, Any]:
        """Validate a citation string and return a structured verification report."""
        decomposed = extract_citation_details(citation)

        if "Error" in decomposed:
            return {
                "valid": False,
                "stage": "parsing",
                "error": decomposed["Error"],
                "parsed": decomposed,
                "act_found": False,
                "section_found": False,
                "llm_result": None,
            }

        act_name = decomposed["Act_Name"]
        year = decomposed["Year"]
        section_number = decomposed.get("Section")

        response = self.section_fetcher(
            act_name=act_name,
            year=year,
            section=section_number,
        )

        is_act_found = bool(response.get("act-found"))
        is_section_found = bool(response.get("section-found"))

        if not (is_act_found and is_section_found):
            return {
                "valid": False,
                "stage": "lookup",
                "error": response.get("error-massage", "Act or section not found."),
                "parsed": decomposed,
                "act_found": is_act_found,
                "section_found": is_section_found,
                "llm_result": None,
            }

        context = _build_act_context(act_name, response.get("sections", []))
        llm_result = self.llm_validator(context, citation)

        return {
            "valid": True,
            "stage": "completed",
            "error": None,
            "parsed": decomposed,
            "act_found": is_act_found,
            "section_found": is_section_found,
            "context": context,
            "llm_result": llm_result,
        }


def validate_citation(citation: str) -> dict[str, Any]:
    """Validate a single legal citation using default database and LLM clients."""
    return CitationValidator().validate(citation)


if __name__ == "__main__":
    sample = (   
        "The Carriage of Goods by Sea Act, 2025 Every notification issued under "
        "this Act shall be laid, as soon as may be after it is issued, before each "
        "House of Parliament, while it is in session, for a total period of thirty "
        "days which may be comprised"
    )
    print(validate_citation(sample))
