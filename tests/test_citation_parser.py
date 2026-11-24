"""Unit tests for barrister_ai.citation_validation.parser."""

from __future__ import annotations

import unittest

from barrister_ai.citation_validation.parser import (
    ParsedCitation,
    extract_citation_details,
    parse_citation,
)


class TestCitationParser(unittest.TestCase):
    def test_extracts_act_year_section_and_subsection(self) -> None:
        text = "Under Section 138(1)(b) of the Negotiable Instruments Act, 1881, a notice is required."
        parsed = parse_citation(text)
        self.assertIsInstance(parsed, ParsedCitation)
        self.assertEqual(parsed.act_name, "Negotiable Instruments Act")
        self.assertEqual(parsed.year, "1881")
        self.assertEqual(parsed.section, "138")
        self.assertEqual(parsed.sub_section, "(1)(b)")
        self.assertIn("notice is required", parsed.proposition)

    def test_extracts_without_section(self) -> None:
        text = (
            "The Carriage of Goods by Sea Act, 2025 Every notification issued "
            "shall be laid before Parliament."
        )
        details = extract_citation_details(text)
        self.assertEqual(details["Act_Name"], "The Carriage of Goods by Sea Act")
        self.assertEqual(details["Year"], "2025")
        self.assertIsNone(details["Section"])
        self.assertIsNone(details["Sub-section"])
        self.assertNotIn("Error", details)

    def test_returns_error_when_act_or_year_missing(self) -> None:
        text = "Section 42 requires a notice within thirty days."
        details = extract_citation_details(text)
        self.assertIn("Error", details)
        self.assertEqual(details["Original_Text"], text)

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            parse_citation("   ")


if __name__ == "__main__":
    unittest.main()
