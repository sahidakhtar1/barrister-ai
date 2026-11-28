"""Citation validation subpackage for Barrister_AI."""

from barrister_ai.citation_validation.llm_client import (
    CitationLLMClient,
    validate_citation_locally,
)
from barrister_ai.citation_validation.parser import (
    ParsedCitation,
    extract_citation_details,
    parse_citation,
)
from barrister_ai.citation_validation.validator import (
    CitationValidator,
    validate_citation,
)

__all__ = [
    "CitationLLMClient",
    "CitationValidator",
    "ParsedCitation",
    "extract_citation_details",
    "parse_citation",
    "validate_citation",
    "validate_citation_locally",
]
