"""LLM client wrapper for grounding citation validation against Act context."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from barrister_ai.config import LLMConfig

SYSTEM_INSTRUCTION = (
    "You are a strict legal citation validator. Your task is to verify if the provided 'Citation' \n"
    "is accurately supported by and valid according to the provided 'Act Context'. \n"
    "Do not use any external knowledge. Base your decision entirely on the provided text.\n\n"
    "Output your response in JSON format in the following structure:\n"
    "- Status: [Valid / Invalid / Partially Valid]\n"
    "- Explanation: [Provide a brief, factual explanation referencing the specific section in the context.]"
)


class CitationLLMClient:
    """OpenAI-compatible client for local or hosted citation verification."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config or LLMConfig()
        self._client = client

    @property
    def client(self) -> Any:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    def validate(
        self,
        act_context: str,
        citation: str,
        prompt_details: str = "",
        model: str | None = None,
    ) -> str:
        """Verify whether *citation* is supported by *act_context*."""
        user_prompt = (
            f"Act Context:\n    {act_context}\n\n"
            f"    Citation/Claim to Validate:\n    {citation}\n\n"
            f"    Additional Instructions:\n    {prompt_details}"
        )

        response = self.client.chat.completions.create(
            model=model or self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


def validate_citation_locally(
    act_context: str,
    citation: str,
    prompt_details: str = "",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Functional helper to validate a citation against Act context via LLM."""
    default_cfg = LLMConfig()
    cfg = LLMConfig(
        model=model or default_cfg.model,
        base_url=base_url or default_cfg.base_url,
        api_key=api_key or default_cfg.api_key,
    )
    return CitationLLMClient(config=cfg).validate(
        act_context=act_context,
        citation=citation,
        prompt_details=prompt_details,
    )
