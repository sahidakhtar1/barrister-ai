"""Unit tests for barrister_ai.citation_validation.validator."""

from __future__ import annotations

import unittest

from barrister_ai.citation_validation.validator import CitationValidator


class TestCitationValidator(unittest.TestCase):
    def test_valid_citation_runs_llm_with_built_context(self) -> None:
        captured: dict[str, str] = {}

        def fake_fetcher(act_name: str, year: str, section: str | None = None):
            return {
                "act-found": True,
                "section-found": True,
                "sections": [
                    {
                        "section_number": "10",
                        "section_title": "Laying of notification",
                        "sub-section": "Every notification issued shall be laid before Parliament.",
                    }
                ],
            }

        def fake_llm(act_context: str, citation: str) -> str:
            captured["context"] = act_context
            captured["citation"] = citation
            return '{"Status": "Valid", "Explanation": "Matches Section 10."}'

        validator = CitationValidator(
            section_fetcher=fake_fetcher,
            llm_validator=fake_llm,
        )
        result = validator.validate(
            "Under Section 10 of The Carriage of Goods by Sea Act, 2025, notifications are laid before Parliament."
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["stage"], "completed")
        self.assertTrue(result["act_found"])
        self.assertTrue(result["section_found"])
        self.assertIn("Section number: 10", captured["context"])
        self.assertIn("Valid", result["llm_result"])

    def test_parse_error_stops_before_lookup(self) -> None:
        called = False

        def fail_if_called(**_kwargs):
            nonlocal called
            called = True
            return {}

        validator = CitationValidator(section_fetcher=fail_if_called)
        result = validator.validate("No act or year here")

        self.assertFalse(result["valid"])
        self.assertEqual(result["stage"], "parsing")
        self.assertFalse(called)

    def test_missing_section_stops_before_llm(self) -> None:
        llm_called = False

        def fake_fetcher(**_kwargs):
            return {
                "act-found": True,
                "section-found": False,
                "error-massage": "Section 999 not found",
            }

        def fake_llm(_ctx: str, _cit: str) -> str:
            nonlocal llm_called
            llm_called = True
            return ""

        validator = CitationValidator(
            section_fetcher=fake_fetcher,
            llm_validator=fake_llm,
        )
        result = validator.validate(
            "Section 999 of The Coastal Shipping Act, 2025 applies."
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["stage"], "lookup")
        self.assertTrue(result["act_found"])
        self.assertFalse(result["section_found"])
        self.assertFalse(llm_called)


if __name__ == "__main__":
    unittest.main()
