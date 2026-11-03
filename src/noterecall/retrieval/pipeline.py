"""The staged retrieval pipeline. Implements ADR-005 (hybrid + RRF) and ADR-006 (rerank).

A `PipelineSpec` names one point of the ablation: which lexical tokenizer (if any), which dense
model (if any), the RRF k when both are used, and whether to rerank. `RetrievalPipeline` runs it:
each retriever returns `candidate_depth` hits, hybrid configurations fuse them with RRF, the
first `fused_depth` hits go on (to the cross-encoder when reranking), and `fused_depth` hits are
returned, best first. Metrics take their own prefix of that list. Per-stage wall-clock times
are measured on every call, because the reranker is only adopted within the latency budget.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from noterecall.index.lexical import LexicalIndex
from noterecall.models import Chunk, RetrievalHit
from noterecall.retrieval.dense import DenseIndex
from noterecall.retrieval.fusion import rrf
from noterecall.retrieval.rerank import Reranker, rerank


class PipelineSpec(BaseModel):
    """One retrieval configuration. `rrf_k` is set exactly when both retrievers are used."""

    model_config = ConfigDict(frozen=True)

    lexical: Literal["plain", "code"] | None = None
    dense_model: str | None = None
    rrf_k: int | None = None
    rerank: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        hybrid = self.lexical is not None and self.dense_model is not None
        if self.lexical is None and self.dense_model is None:
            raise ValueError("a pipeline needs at least one retriever")
        if hybrid != (self.rrf_k is not None):
            raise ValueError("rrf_k must be set exactly when both retrievers are used")
        return self


class Variant(BaseModel):
    """What the indexed text is made of (ADR-007): with `header`, the deterministic context
    header is put in front of each chunk. Off is the M4 pipeline."""

    model_config = ConfigDict(frozen=True)

    header: bool = False


class Depths(BaseModel):
    """How many hits each stage passes on."""

    model_config = ConfigDict(frozen=True)

    candidate: int
    fused: int


class StageTimes(BaseModel):
    """Wall-clock milliseconds per stage for one query (0 for a stage that did not run)."""

    model_config = ConfigDict(frozen=True)

    lexical_ms: float = 0.0
    dense_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class Components:
    """The built indexes and models a pipeline may use."""

    lexical: LexicalIndex | None = None
    dense: DenseIndex | None = None
    reranker: Reranker | None = None


class PipelineError(Exception):
    """Raised when a spec asks for a component that was not provided."""


class RetrievalPipeline:
    """A built pipeline: indexes and models are supplied, queries are cheap."""

    def __init__(
        self,
        spec: PipelineSpec,
        chunks: Mapping[str, Chunk],
        depths: Depths,
        components: Components,
    ) -> None:
        if (spec.lexical is None) != (components.lexical is None):
            raise PipelineError("the lexical index must be given exactly when spec.lexical is set")
        if (spec.dense_model is None) != (components.dense is None):
            raise PipelineError(
                "the dense index must be given exactly when spec.dense_model is set"
            )
        if spec.rerank and components.reranker is None:
            raise PipelineError("spec.rerank is set but no reranker was given")
        self.spec = spec
        self._chunks = chunks
        self._depths = depths
        self._lexical = components.lexical
        self._dense = components.dense
        self._reranker = components.reranker

    def retrieve(self, query: str) -> tuple[list[RetrievalHit], StageTimes]:
        """Ranked hits for the query and how long each stage took."""
        start = time.perf_counter()
        lexical_hits, lexical_ms = self._timed(self._lexical, query)
        dense_hits, dense_ms = self._timed(self._dense, query)
        mark = time.perf_counter()
        if self.spec.rrf_k is not None:
            fused = rrf(
                [[h.chunk_id for h in lexical_hits], [h.chunk_id for h in dense_hits]],
                self.spec.rrf_k,
            )
        else:
            fused = lexical_hits or dense_hits
        hits = fused[: self._depths.fused]
        fusion_ms = (time.perf_counter() - mark) * 1000
        rerank_ms = 0.0
        if self._reranker is not None and self.spec.rerank:
            mark = time.perf_counter()
            hits = rerank(query, hits, self._chunks, self._reranker, self._depths.fused)
            rerank_ms = (time.perf_counter() - mark) * 1000
        times = StageTimes(
            lexical_ms=lexical_ms,
            dense_ms=dense_ms,
            fusion_ms=fusion_ms,
            rerank_ms=rerank_ms,
            total_ms=(time.perf_counter() - start) * 1000,
        )
        return hits, times

    def _timed(
        self, index: LexicalIndex | DenseIndex | None, query: str
    ) -> tuple[list[RetrievalHit], float]:
        if index is None:
            return [], 0.0
        mark = time.perf_counter()
        hits = index.search(query, self._depths.candidate)
        return hits, (time.perf_counter() - mark) * 1000
