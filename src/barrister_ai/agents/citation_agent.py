"""Citation Validation Agent.

Receives a user query whose intent is ``citation_validation`` and
orchestrates the full citation-verification pipeline:
  1. Parse the citation (regex-based)
  2. Look up act/section in PostgreSQL
  3. Confirm via local LLM
"""

from __future__ import annotations

import logging
from typing import Any

from barrister_ai.citation_validation.validator import CitationValidator

logger = logging.getLogger(__name__)


class CitationValidationAgent:
    """Agent responsible for validating legal citations.

    Parameters
    ----------
    section_fetcher:
        Optional override for the database section-lookup callable.
        Defaults to the production PostgreSQL repository.
    llm_validator:
        Optional override for the LLM confirmation callable.
        Defaults to the local LLM client.
    """

    AGENT_NAME = "CitationValidationAgent"
    SUPPORTED_INTENT = "citation_validation"

    def __init__(
        self,
        section_fetcher: Any | None = None,
        llm_validator: Any | None = None,
    ) -> None:
        self._validator = CitationValidator(
            section_fetcher=section_fetcher,
            llm_validator=llm_validator,
        )

    def run(self, query: str, intent_result: dict[str, Any]) -> dict[str, Any]:
        """Execute citation validation for *query*.

        Parameters
        ----------
        query:
            Original user query that contains or describes a legal citation.
        intent_result:
            Output dict from :func:`~barrister_ai.intent.classifier.classify_intent`.

        Returns
        -------
        dict with keys ``agent``, ``intent``, ``query``, ``answer``,
        ``validation``.
        """
        logger.info("%s: processing query=%r", self.AGENT_NAME, query)

        try:
            validation = self._validator.validate(query)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s: validation error – %s", self.AGENT_NAME, exc)
            validation = {
                "valid": False,
                "stage": "error",
                "error": str(exc),
            }

        if validation.get("valid"):
            parsed = validation.get("parsed", {})
            act = parsed.get("Act_Name", "")
            year = parsed.get("Year", "")
            section = parsed.get("Section", "")
            llm_verdict = validation.get("llm_result", "No LLM verdict available.")
            answer = (
                f"✅ Citation validated successfully.\n"
                f"  Act : {act} ({year})\n"
                f"  Section : {section}\n\n"
                f"LLM Verification:\n{llm_verdict}"
            )
        else:
            stage = validation.get("stage", "unknown")
            error = validation.get("error", "Unknown error.")
            act_found = validation.get("act_found", False)
            sec_found = validation.get("section_found", False)
            answer = (
                f"❌ Citation could not be validated (stage: {stage}).\n"
                f"  Act found in database  : {'Yes' if act_found else 'No'}\n"
                f"  Section found          : {'Yes' if sec_found else 'No'}\n"
                f"  Reason : {error}"
            )

        return {
            "agent": self.AGENT_NAME,
            "intent": self.SUPPORTED_INTENT,
            "query": query,
            "answer": answer,
            "validation": validation,
        }
