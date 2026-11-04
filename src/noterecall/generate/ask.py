"""Asking a question end to end. Implements ADR-011 and proposal 006.

`Asker.ask` retrieves with the decided pipeline, applies stage 1 of abstention to the two signals
(a refusal makes no LLM call at all), and otherwise gives the top `max_chunks` chunks to the
`Generator`, whose own validation and `insufficient` flag are stage 2.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from noterecall.generate.abstain import Thresholds, refuse
from noterecall.generate.answer import Generator, Outcome, Response
from noterecall.generate.verify import Verifier
from noterecall.models import Chunk
from noterecall.retrieval.pipeline import AbstentionSignals, RetrievalPipeline


class AskResult(BaseModel):
    """The final response, the signals stage 1 saw, and the chunk ids that were in the context.
    `drafted` is the response before stage 3 (the same object when stage 3 is off)."""

    model_config = ConfigDict(frozen=True)

    response: Response
    drafted: Response
    signals: AbstentionSignals
    context_ids: tuple[str, ...]


@dataclass(frozen=True)
class Models:
    """The model-side stages: the answer generator and, if stage 3 is on, the verifier."""

    generator: Generator
    verifier: Verifier | None = None


class Asker:
    """Retrieval, stage-1 abstention and generation behind one call."""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        chunks: Mapping[str, Chunk],
        thresholds: Thresholds,
        models: Models,
        max_chunks: int,
    ) -> None:
        self._pipeline = pipeline
        self._chunks = chunks
        self._thresholds = thresholds
        self._models = models
        self._max_chunks = max_chunks

    @property
    def models(self) -> Models:
        return self._models

    @property
    def generator(self) -> Generator:
        return self._models.generator

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
            refusal = Response(outcome=Outcome.REFUSED_STAGE1)
            return AskResult(response=refusal, drafted=refusal, signals=signals, context_ids=ids)
        drafted = self._models.generator.answer(question, chunks)
        final = drafted
        if self._models.verifier is not None:
            final = self._models.verifier.verify(question, drafted, self._chunks)
        return AskResult(response=final, drafted=drafted, signals=signals, context_ids=ids)
