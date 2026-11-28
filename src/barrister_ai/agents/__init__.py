"""Specialised legal processing agents for Barrister_AI."""

from barrister_ai.agents.summarisation_agent import SummarisationAgent
from barrister_ai.agents.citation_agent import CitationValidationAgent
from barrister_ai.agents.precedent_agent import PrecedentSearchAgent

__all__ = [
    "SummarisationAgent",
    "CitationValidationAgent",
    "PrecedentSearchAgent",
]
