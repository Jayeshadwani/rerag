"""The citation validator, the anti-hallucination gate. Implements ADR-011 (grounding in code).

Chunks go into the prompt under short IDs (`C1`, `C2`, ...). The model must answer in the schema
`Answer`: a required `insufficient` flag and a list of claims, each with the IDs it cites.
`validate_answer` is a pure function that returns every structural problem: an ID that is not in
the context, a claim with no citation, an answer with no claims although it is not insufficient, and
claims given although it says it is insufficient. It checks structure only. Whether a cited chunk
really supports the claim is judged in M7 (ADR-012), and this module does not pretend otherwise.
"""

from collections.abc import Collection
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    """One statement of the answer and the context IDs it cites."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    citations: list[str]


class Answer(BaseModel):
    """The model's answer. `insufficient` true means: the notes do not contain the answer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    insufficient: bool
    claims: list[Claim]


class ProblemKind(StrEnum):
    """What is wrong with an answer."""

    UNKNOWN_ID = "unknown_id"
    EMPTY_CITATIONS = "empty_citations"
    NO_CLAIMS = "no_claims"
    CLAIMS_WHEN_INSUFFICIENT = "claims_when_insufficient"
    UNPARSABLE = "unparsable"


class Problem(BaseModel):
    """One problem, with a message that is sent back to the model on the retry."""

    model_config = ConfigDict(frozen=True)

    kind: ProblemKind
    detail: str

    def message(self) -> str:
        return f"{self.kind.value}: {self.detail}"


def validate_answer(answer: Answer, valid_ids: Collection[str]) -> list[Problem]:
    """Every structural problem of the answer; an empty list means it is valid."""
    if answer.insufficient:
        if answer.claims:
            return [
                Problem(
                    kind=ProblemKind.CLAIMS_WHEN_INSUFFICIENT,
                    detail="insufficient is true, so claims must be empty",
                )
            ]
        return []
    if not answer.claims:
        return [
            Problem(
                kind=ProblemKind.NO_CLAIMS,
                detail="insufficient is false, so at least one claim is required",
            )
        ]
    problems = []
    for number, claim in enumerate(answer.claims, start=1):
        if not claim.citations:
            problems.append(
                Problem(
                    kind=ProblemKind.EMPTY_CITATIONS,
                    detail=f"claim {number} cites nothing",
                )
            )
        problems.extend(
            Problem(
                kind=ProblemKind.UNKNOWN_ID,
                detail=f"claim {number} cites {cited!r}, which is not in the context",
            )
            for cited in claim.citations
            if cited not in valid_ids
        )
    return problems
