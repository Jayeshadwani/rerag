"""Stage 3, post-answer verification. Implements ADR-011 as amended by proposal 007.

After an answer has passed the citation validator, `Verifier.verify` makes one more call (the same
model, a higher reasoning effort) with the question, the numbered draft claims and the full text of
the chunks they cite. The verdict has `answers_question` (do the cited chunks directly answer the
question asked, not a related topic?) and a `supported` flag for every claim. The answer is refused
if `answers_question` is false, unsupported claims are dropped, and it is refused if none remain.
A verdict that does not judge every claim exactly once is retried once and then counts as a
refusal: an answer that could not be verified is not shown. The prompt is `llm/prompts/verify.md`.
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from noterecall.generate.answer import CitedClaim, Outcome, Response, Verification
from noterecall.generate.citations import Problem, ProblemKind
from noterecall.ingest.secrets import scrub
from noterecall.llm.client import LLMClient, LLMOutputError, Prompt, Usage
from noterecall.llm.prompts import PromptTemplate
from noterecall.models import Chunk

MAX_PROBLEM_CHARS = 300


class ClaimVerdict(BaseModel):
    """The verdict on one claim, by its number in the draft (1-based)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: int = Field(ge=1)
    supported: bool


class Verdict(BaseModel):
    """The verifier's answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answers_question: bool
    claims: list[ClaimVerdict]


def validate_verdict(verdict: Verdict, claim_count: int) -> list[Problem]:
    """Every claim number from 1 to `claim_count` must be judged exactly once, and no other."""
    problems = []
    seen: dict[int, int] = {}
    for entry in verdict.claims:
        seen[entry.claim] = seen.get(entry.claim, 0) + 1
    for number in sorted(seen):
        if not 1 <= number <= claim_count:
            detail = f"claim {number} does not exist (the draft has {claim_count})"
            problems.append(Problem(kind=ProblemKind.BAD_VERDICT, detail=detail))
        elif seen[number] > 1:
            detail = f"claim {number} was judged {seen[number]} times"
            problems.append(Problem(kind=ProblemKind.BAD_VERDICT, detail=detail))
    problems.extend(
        Problem(kind=ProblemKind.BAD_VERDICT, detail=f"claim {number} was not judged")
        for number in range(1, claim_count + 1)
        if number not in seen
    )
    return problems


def apply_verdict(
    verdict: Verdict, claims: Sequence[CitedClaim]
) -> tuple[Outcome, tuple[CitedClaim, ...]]:
    """The outcome and the claims that are kept. Needs a valid verdict."""
    if not verdict.answers_question:
        return Outcome.REFUSED_VERIFIER, ()
    supported = {v.claim for v in verdict.claims if v.supported}
    kept = tuple(c for n, c in enumerate(claims, start=1) if n in supported)
    return (Outcome.ANSWERED, kept) if kept else (Outcome.REFUSED_UNSUPPORTED, ())


def format_claims(claims: Sequence[CitedClaim], numbering: Mapping[str, str]) -> str:
    """The numbered claims with the (relabelled) notes each one cites."""
    lines = []
    for n, claim in enumerate(claims, start=1):
        cites = ", ".join(numbering[cid] for cid in claim.chunk_ids)
        lines.append(f"{n}. {claim.text} (cites: {cites})")
    return "\n".join(lines)


def cited_notes(
    claims: Sequence[CitedClaim], chunks: Mapping[str, Chunk]
) -> tuple[dict[str, str], str]:
    """The cited chunks only, labelled N1, N2, ... by first citation, each with where it is from."""
    numbering: dict[str, str] = {}
    for claim in claims:
        for cid in claim.chunk_ids:
            numbering.setdefault(cid, f"N{len(numbering) + 1}")
    blocks = []
    for cid, label in numbering.items():
        chunk = chunks[cid]
        where = " > ".join([chunk.notebook_path, *chunk.heading_path])
        blocks.append(f"[{label}] {where}\n{chunk.text}")
    return numbering, "\n\n".join(blocks)


def feedback_text(problems: Sequence[Problem]) -> str:
    """The retry message appended to the prompt."""
    lines = "\n".join(f"- {p.message()}" for p in problems)
    return (
        "\n\n<problems_with_your_previous_verdict>\n"
        f"{lines}\n"
        "</problems_with_your_previous_verdict>\n"
        "Answer again and judge every claim exactly once."
    )


class Verifier:
    """Stage 3 behind one call."""

    def __init__(
        self,
        client: LLMClient,
        template: PromptTemplate,
        model: str,
        settings: tuple[str, int],
    ) -> None:
        self._client = client
        self._template = template
        self._model = model
        self._effort, self._max_output_tokens = settings

    @property
    def prompt_version(self) -> str:
        return self._template.version

    def prompt_for(
        self,
        question: str,
        claims: Sequence[CitedClaim],
        chunks: Mapping[str, Chunk],
        feedback: str = "",
    ) -> Prompt:
        """The verifier prompt for a draft answer (also used to estimate cost)."""
        numbering, notes = cited_notes(claims, chunks)
        return self._template.render(
            self._model,
            self._max_output_tokens,
            reasoning_effort=self._effort,
            question=scrub(question).text,
            claims=format_claims(claims, numbering),
            notes=notes,
            feedback=feedback,
        )

    def verify(self, question: str, response: Response, chunks: Mapping[str, Chunk]) -> Response:
        """Stage 3 for an answered response; any other response is returned unchanged."""
        if response.outcome is not Outcome.ANSWERED:
            return response
        usage, latency, feedback = Usage(), 0.0, ""
        for attempt in (1, 2):
            prompt = self.prompt_for(question, response.claims, chunks, feedback)
            try:
                result = self._client.complete_json(prompt, Verdict)
            except LLMOutputError as exc:
                problems = [
                    Problem(kind=ProblemKind.UNPARSABLE, detail=str(exc)[:MAX_PROBLEM_CHARS])
                ]
            else:
                usage, latency = usage.plus(result.usage), latency + result.api_latency_s
                problems = validate_verdict(result.parsed, len(response.claims))
                if not problems:
                    outcome, kept = apply_verdict(result.parsed, response.claims)
                    note = Verification(
                        answers_question=result.parsed.answers_question,
                        claims_in=len(response.claims),
                        claims_dropped=len(response.claims) - len(kept),
                        attempts=attempt,
                        usage=usage,
                        api_latency_s=latency,
                    )
                    return response.model_copy(
                        update={"outcome": outcome, "claims": kept, "verification": note}
                    )
            feedback = feedback_text(problems)
        note = Verification(answers_question=None, claims_in=len(response.claims),
                            claims_dropped=len(response.claims), attempts=2, usage=usage,
                            api_latency_s=latency)  # fmt: skip
        return response.model_copy(
            update={"outcome": Outcome.REFUSED_VERIFIER_INVALID, "claims": (), "verification": note}
        )
