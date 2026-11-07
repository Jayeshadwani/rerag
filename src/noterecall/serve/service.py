"""The question-answering service behind the API. Implements ADR-011 (grounding) for the MVP.

`QAService.ask` keeps the last few turns of the history, rewrites a follow-up into a standalone
question when there is history (one small model call; none otherwise), and runs the normal
pipeline on that question: retrieve, grounded answer with validated citations, refusal. The answer
step never sees the history, so only retrieved chunks can be cited. The response names each cited
chunk `C1`, `C2`, ... after its position in the context and carries those chunks as sources."""

from collections.abc import Mapping, Sequence

from noterecall.api_models import (
    AskRequest,
    AskResponse,
    Claim,
    RefusalReason,
    Source,
    Status,
    Turn,
)
from noterecall.config import ApiConfig
from noterecall.generate.answer import Outcome
from noterecall.generate.ask import Asker, AskResult
from noterecall.generate.rewrite import Rewriter
from noterecall.models import Chunk

REASONS = {
    Outcome.REFUSED_STAGE1: RefusalReason.BELOW_THRESHOLD,
    Outcome.REFUSED_MODEL: RefusalReason.NOT_IN_NOTES,
    Outcome.REFUSED_INVALID: RefusalReason.INVALID_ANSWER,
}


def recent_turns(history: Sequence[Turn], keep: int, max_chars: int) -> list[Turn]:
    """The last `keep` turns, each text cut to `max_chars`."""
    if keep == 0:
        return []
    return [
        Turn(question=t.question[:max_chars], answer=t.answer[:max_chars]) for t in history[-keep:]
    ]


def to_response(standalone: str, result: AskResult, chunks: Mapping[str, Chunk]) -> AskResponse:
    """The API response for a pipeline result."""
    response = result.response
    if response.outcome is not Outcome.ANSWERED:
        return AskResponse(
            status=Status.REFUSED,
            refusal_reason=REASONS[response.outcome],
            standalone_question=standalone,
            claims=[],
            sources=[],
        )
    short = {cid: f"C{n}" for n, cid in enumerate(result.context_ids, start=1)}
    claims = [
        Claim(text=c.text, citations=[short[i] for i in c.chunk_ids]) for c in response.claims
    ]
    cited = sorted(
        {short[i] for c in response.claims for i in c.chunk_ids}, key=lambda s: int(s[1:])
    )
    by_short = {v: k for k, v in short.items()}
    sources = [source(label, chunks[by_short[label]]) for label in cited]
    return AskResponse(
        status=Status.ANSWERED,
        refusal_reason=None,
        standalone_question=standalone,
        claims=claims,
        sources=sources,
    )


def source(label: str, chunk: Chunk) -> Source:
    """A cited chunk with its place in the notebook."""
    first_cell = chunk.pieces[0].cell_index if chunk.pieces else 0
    return Source(
        id=label,
        notebook=chunk.notebook_path,
        heading_path=list(chunk.heading_path),
        cell_index=first_cell,
        text=chunk.text,
    )


class QAService:
    """History handling, the optional rewrite, and the pipeline."""

    def __init__(
        self, asker: Asker, rewriter: Rewriter, chunks: Mapping[str, Chunk], limits: ApiConfig
    ) -> None:
        self._asker = asker
        self._rewriter = rewriter
        self._chunks = chunks
        self._limits = limits

    def ask(self, request: AskRequest) -> AskResponse:
        question = request.question.strip()
        history = recent_turns(
            request.history, self._limits.history_turns, self._limits.max_turn_chars
        )
        standalone = self._rewriter.rewrite(question, history) if history else question
        return to_response(standalone, self._asker.ask(standalone), self._chunks)
