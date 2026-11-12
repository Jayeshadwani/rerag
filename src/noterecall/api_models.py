"""The request and response shapes of the HTTP service. Implements proposal 009 (the API contract).

`POST /ask` takes a question and the last turns of the browser session's conversation and returns
either an answer (claims with citations such as `C1`, plus the cited sources) or a refusal with a
reason. The history is only used to rewrite a follow-up into a standalone question; it is never
evidence. These models are the only contract between the API and the UI."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    """One earlier exchange of the conversation: the question asked and the answer text shown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    answer: str


class AskRequest(BaseModel):
    """A question and the recent history (the service keeps only the last few turns)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1)
    history: list[Turn] = []


class Status(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"


class RefusalReason(StrEnum):
    """Why there is no answer."""

    NOT_IN_NOTES = "not_in_notes"  # the model found nothing in the retrieved notes that answers it
    INVALID_ANSWER = "invalid_answer"  # its citations failed validation twice
    BELOW_THRESHOLD = "below_threshold"  # stage 1: the best match was too weak (no model call)


class Claim(BaseModel):
    """One statement of the answer and the ids of the sources it cites (`C1`, `C2`, ...)."""

    model_config = ConfigDict(frozen=True)

    text: str
    citations: list[str]


class Source(BaseModel):
    """A retrieved chunk that an answer cites, with its location in the notebook."""

    model_config = ConfigDict(frozen=True)

    id: str
    notebook: str
    heading_path: list[str]
    cell_index: int  # the first cell of the chunk
    text: str


class AskResponse(BaseModel):
    """The result of a question."""

    model_config = ConfigDict(frozen=True)

    status: Status
    refusal_reason: RefusalReason | None
    standalone_question: str
    claims: list[Claim]
    sources: list[Source]


class TopicsResponse(BaseModel):
    """`GET /topics`: the notebook titles in the index, by notebook number. No note content."""

    model_config = ConfigDict(frozen=True)

    count: int
    notebooks: list[str]


class ErrorBody(BaseModel):
    """The body of an error status (429 and 5xx): a short, generic message."""

    model_config = ConfigDict(frozen=True)

    detail: str
    retry_after_s: int | None = None
