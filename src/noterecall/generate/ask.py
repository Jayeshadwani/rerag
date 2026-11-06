"""Asking a question end to end. Implements ADR-011 and proposal 006.

`Asker.ask` retrieves with the decided pipeline, applies stage 1 of abstention to the two signals
(a refusal makes no LLM call at all), and otherwise gives the top `max_chunks` chunks to the
`Generator`, whose own validation and `insufficient` flag are stage 2.
"""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from noterecall.generate.abstain import Thresholds, refuse
from noterecall.generate.answer import Generator, Outcome, Response
from noterecall.models import Chunk
from noterecall.retrieval.pipeline import AbstentionSignals, RetrievalPipeline


class AskResult(BaseModel):
    """The response, the signals stage 1 saw, and the chunk ids that were in the context."""

    model_config = ConfigDict(frozen=True)

    response: Response
    signals: AbstentionSignals
    context_ids: tuple[str, ...]


class Asker:
    """Retrieval, stage-1 abstention and generation behind one call."""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        chunks: Mapping[str, Chunk],
        thresholds: Thresholds,
        generator: Generator,
        max_chunks: int,
    ) -> None:
        self._pipeline = pipeline
        self._chunks = chunks
        self._thresholds = thresholds
        self._generator = generator
        self._max_chunks = max_chunks

    @property
    def generator(self) -> Generator:
        return self._generator

    def context_for(self, question: str) -> tuple[list[Chunk], AbstentionSignals]:
        """The chunks that would go into the prompt, and the stage-1 signals."""
        hits, signals = self._pipeline.retrieve_with_signals(question)
        return [self._chunks[h.chunk_id] for h in hits[: self._max_chunks]], signals

    def stage1_refuses(self, signals: AbstentionSignals) -> bool:
        return refuse(signals, self._thresholds)

    def ask(self, question: str) -> AskResult:
        chunks, signals = self.context_for(question)
        ids = tuple(c.id for c in chunks)
        if self.stage1_refuses(signals):
            return AskResult(
                response=Response(outcome=Outcome.REFUSED_STAGE1), signals=signals, context_ids=ids
            )
        response = self._generator.answer(question, chunks)
        return AskResult(response=response, signals=signals, context_ids=ids)
