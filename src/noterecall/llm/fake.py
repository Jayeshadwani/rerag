"""Deterministic offline stand-in for the LLM transport (CLAUDE.md: tests use `llm/fake.py`).

`FakeTransport` answers each prompt from a function of the prompt, records every prompt it was
given (so tests can check what would have been sent, for example that a planted secret never
reaches the provider), and reports usage as word counts and, if given a latency function, that
latency. It can be told to fail.
"""

from collections.abc import Callable

from noterecall.llm.client import LLMCallError, Prompt, Usage
from noterecall.llm.transport import RawCompletion

Responder = Callable[[Prompt], str]
Latency = Callable[[Prompt], float]


def count_words(text: str) -> int:
    """The fake's notion of a token."""
    return len(text.split())


class FakeTransport:
    """Answers with `responder(prompt)` (the JSON text) and records every prompt."""

    def __init__(
        self,
        responder: Responder,
        fail_after: int | None = None,
        latency: Latency | None = None,
    ) -> None:
        self._responder = responder
        self._fail_after = fail_after
        self._latency = latency
        self.prompts: list[Prompt] = []

    def complete_text(
        self, prompt: Prompt, schema_name: str, schema_json: dict[str, object]
    ) -> RawCompletion:
        if self._fail_after is not None and len(self.prompts) >= self._fail_after:
            raise LLMCallError("the fake was told to fail")
        self.prompts.append(prompt)
        text = self._responder(prompt)
        sent = sum(
            count_words(part) for part in (prompt.instructions, prompt.shared, prompt.variable)
        )
        usage = Usage(input_tokens=sent, output_tokens=count_words(text))
        latency = self._latency(prompt) if self._latency is not None else 0.0
        return RawCompletion(text=text, usage=usage, latency_s=latency)
