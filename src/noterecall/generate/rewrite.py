"""Rewriting a follow-up question into a standalone one. Implements proposal 009 (conversation
memory).

With history, one small model call turns "and why?" into a question that can be understood on its
own; the normal pipeline (retrieve, grounded answer, refusal) then runs on that question. The
history is only read here, to resolve references: it is never given to the answer step, so it can
never be evidence or be cited. Question and history are secret-scrubbed first. An unusable rewrite
raises `RewriteError`: there is no silent fallback to the original question. The prompt is
`llm/prompts/rewrite_question.md`."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from noterecall.api_models import Turn
from noterecall.ingest.secrets import scrub
from noterecall.llm.client import LLMClient, LLMOutputError
from noterecall.llm.prompts import PromptTemplate


class RewriteError(Exception):
    """Raised when the follow-up could not be turned into a usable standalone question."""


class Standalone(BaseModel):
    """The model's answer: one standalone question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str


def format_history(history: Sequence[Turn], max_chars: int) -> str:
    """The earlier turns as text, each text cut to `max_chars`, secrets scrubbed."""
    blocks = []
    for number, turn in enumerate(history, start=1):
        asked = scrub(turn.question[:max_chars]).text
        shown = scrub(turn.answer[:max_chars]).text
        blocks.append(f"Turn {number}\nQuestion: {asked}\nAnswer: {shown}")
    return "\n\n".join(blocks)


class Rewriter:
    """One bounded call: follow-up plus history in, standalone question out."""

    def __init__(
        self,
        client: LLMClient,
        template: PromptTemplate,
        model: str,
        limits: tuple[int, int, int],
    ) -> None:
        self._client = client
        self._template = template
        self._model = model
        self._max_output_tokens, self._max_chars, self._max_turn_chars = limits

    def rewrite(self, question: str, history: Sequence[Turn]) -> str:
        """The standalone question for `question` given `history` (which must not be empty)."""
        if not history:
            raise RewriteError("there is no history to rewrite against")
        prompt = self._template.render(
            self._model,
            self._max_output_tokens,
            history=format_history(history, self._max_turn_chars),
            question=scrub(question).text,
        )
        try:
            standalone = self._client.complete_json(prompt, Standalone).parsed.question.strip()
        except LLMOutputError as exc:
            raise RewriteError(f"the rewrite was unusable: {exc}") from exc
        if not standalone:
            raise RewriteError("the rewrite was empty")
        if len(standalone) > self._max_chars:
            raise RewriteError(f"the rewrite is longer than {self._max_chars} characters")
        return standalone
