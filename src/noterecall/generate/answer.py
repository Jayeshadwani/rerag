"""Answer generation with enforced citations. Implements ADR-011 (grounding) and proposal 006
(stage 2: the model's `insufficient` flag).

`Generator.answer` puts the retrieved chunks into the prompt under the IDs C1, C2, ..., asks for an
`Answer` in the strict schema, and validates it with `citations.validate_answer`. If the first
attempt has problems it is asked once more with the problems listed; if that also fails the
response is a refusal, never an unvalidated answer. The question is passed through the secret
scrubber before it is sent. The prompt is `llm/prompts/answer.md`.
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from noterecall.generate.citations import Answer, Problem, ProblemKind, validate_answer
from noterecall.ingest.secrets import scrub
from noterecall.llm.client import LLMClient, LLMOutputError, Prompt, Usage
from noterecall.llm.prompts import PromptTemplate
from noterecall.models import Chunk

MAX_PROBLEM_CHARS = 300


class Outcome(StrEnum):
    """How a question ended."""

    ANSWERED = "answered"
    REFUSED_STAGE1 = "refused_stage1"
    REFUSED_MODEL = "refused_model"
    REFUSED_INVALID = "refused_invalid"
    REFUSED_VERIFIER = "refused_verifier"
    REFUSED_UNSUPPORTED = "refused_unsupported"
    REFUSED_VERIFIER_INVALID = "refused_verifier_invalid"


class CitedClaim(BaseModel):
    """A claim with the chunk ids its short IDs resolve to."""

    model_config = ConfigDict(frozen=True)

    text: str
    chunk_ids: tuple[str, ...]


class FirstAttempt(BaseModel):
    """What the first attempt looked like, for the citation-validity metrics."""

    model_config = ConfigDict(frozen=True)

    parsed: bool
    claims: int
    uncited_claims: int
    unknown_citations: int
    valid: bool


class Verification(BaseModel):
    """What stage 3 did to an answered response (proposal 007)."""

    model_config = ConfigDict(frozen=True)

    answers_question: bool | None
    claims_in: int
    claims_dropped: int
    attempts: int
    usage: Usage
    api_latency_s: float


class Response(BaseModel):
    """The result of asking a question. `usage` and `api_latency_s` are those of the answer calls;
    stage 3, if it ran, has its own in `verification`."""

    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    claims: tuple[CitedClaim, ...] = ()
    attempts: int = 0
    first_attempt: FirstAttempt | None = None
    usage: Usage = Usage()
    api_latency_s: float = 0.0
    verification: Verification | None = None


def short_ids(chunks: Sequence[Chunk]) -> dict[str, Chunk]:
    """C1, C2, ... in rank order."""
    return {f"C{n}": chunk for n, chunk in enumerate(chunks, start=1)}


def format_notes(context: dict[str, Chunk]) -> str:
    """The notes block of the prompt: each chunk under its ID with where it comes from."""
    blocks = []
    for short, chunk in context.items():
        where = " > ".join([chunk.notebook_path, *chunk.heading_path])
        blocks.append(f"[{short}] {where}\n{chunk.text}")
    return "\n\n".join(blocks)


def feedback_text(problems: Sequence[Problem]) -> str:
    """The retry message appended to the prompt."""
    lines = "\n".join(f"- {p.message()}" for p in problems)
    return (
        "\n\n<problems_with_your_previous_answer>\n"
        f"{lines}\n"
        "</problems_with_your_previous_answer>\n"
        "Answer again and fix these problems."
    )


def first_attempt_stats(answer: Answer | None, problems: Sequence[Problem]) -> FirstAttempt:
    """Counts over the first attempt (all zero if it could not be parsed)."""
    if answer is None:
        return FirstAttempt(parsed=False, claims=0, uncited_claims=0, unknown_citations=0,
                            valid=False)  # fmt: skip
    return FirstAttempt(
        parsed=True,
        claims=len(answer.claims),
        uncited_claims=sum(1 for c in answer.claims if not c.citations),
        unknown_citations=sum(1 for p in problems if p.kind is ProblemKind.UNKNOWN_ID),
        valid=not problems,
    )


class Generator:
    """Asks the model and enforces the citation rules."""

    def __init__(
        self, client: LLMClient, template: PromptTemplate, model: str, max_output_tokens: int
    ) -> None:
        self._client = client
        self._template = template
        self._model = model
        self._max_output_tokens = max_output_tokens

    @property
    def prompt_version(self) -> str:
        """The content hash of the answer prompt file."""
        return self._template.version

    def prompt_for(self, question: str, chunks: Sequence[Chunk]) -> Prompt:
        """The first-attempt prompt for a question and its chunks (used to estimate cost)."""
        return self._render(format_notes(short_ids(chunks)), scrub(question).text, "")

    def _render(self, notes: str, question: str, feedback: str) -> Prompt:
        return self._template.render(
            self._model, self._max_output_tokens, notes=notes, question=question, feedback=feedback
        )

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Response:
        """Answer from the chunks, or refuse; at most two model calls."""
        context = short_ids(chunks)
        notes = format_notes(context)
        safe_question = scrub(question).text
        feedback = ""
        usage = Usage()
        latency = 0.0
        first: FirstAttempt | None = None
        for attempt in (1, 2):
            prompt = self._render(notes, safe_question, feedback)
            answer: Answer | None = None
            try:
                result = self._client.complete_json(prompt, Answer)
            except LLMOutputError as exc:
                problems = [
                    Problem(kind=ProblemKind.UNPARSABLE, detail=str(exc)[:MAX_PROBLEM_CHARS])
                ]
            else:
                answer, usage = result.parsed, usage.plus(result.usage)
                latency += result.api_latency_s
                problems = validate_answer(answer, context)
            if attempt == 1:
                first = first_attempt_stats(answer, problems)
            if not problems and answer is not None:
                return self._finish(answer, context, (attempt, first), (usage, latency))
            feedback = feedback_text(problems)
        return Response(
            outcome=Outcome.REFUSED_INVALID,
            attempts=2,
            first_attempt=first,
            usage=usage,
            api_latency_s=latency,
        )

    @staticmethod
    def _finish(
        answer: Answer,
        context: dict[str, Chunk],
        tries: tuple[int, FirstAttempt | None],
        spent: tuple[Usage, float],
    ) -> Response:
        attempts, first = tries
        usage, latency = spent
        common = {"attempts": attempts, "first_attempt": first, "usage": usage,
                  "api_latency_s": latency}  # fmt: skip
        if answer.insufficient:
            return Response(outcome=Outcome.REFUSED_MODEL, **common)
        claims = tuple(
            CitedClaim(text=c.text, chunk_ids=tuple(context[i].id for i in c.citations))
            for c in answer.claims
        )
        return Response(outcome=Outcome.ANSWERED, claims=claims, **common)
