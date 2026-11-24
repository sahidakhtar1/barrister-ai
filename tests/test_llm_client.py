"""Unit tests for barrister_ai.citation_validation.llm_client."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from barrister_ai.citation_validation.llm_client import CitationLLMClient
from barrister_ai.config import LLMConfig


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"Status": "Valid"}')
                )
            ]
        )


class TestCitationLLMClient(unittest.TestCase):
    def test_validate_sends_system_and_user_messages(self) -> None:
        completions = _FakeCompletions()
        fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        client = CitationLLMClient(
            config=LLMConfig(model="test-model", base_url="http://localhost:9999/v1", api_key="k"),
            client=fake_openai,
        )
        output = client.validate(
            act_context="Section 12: Driver license required.",
            citation="Section 12 requires a driver license.",
            prompt_details="Keep brief.",
        )

        self.assertEqual(output, '{"Status": "Valid"}')
        self.assertEqual(completions.last_kwargs["model"], "test-model")
        messages = completions.last_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Section 12: Driver license required.", messages[1]["content"])
        self.assertIn("Keep brief.", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
