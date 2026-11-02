"""Cross-encoder reranking of the fused top candidates. Implements ADR-006.

A `Reranker` scores (query, passage) pairs; `rerank` re-orders the first `top_n` hits by that
score (ties keep their earlier order) and drops the rest. `OverlapReranker` is the
deterministic offline stand-in used by the default tests.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from noterecall.index.code_tokenizer import tokenize_plain
from noterecall.models import Chunk, RetrievalHit


class RerankSpec(BaseModel):
    """The reranker model and its pinned revision."""

    model_config = ConfigDict(frozen=True)

    id: str
    revision: str


class Reranker(Protocol):
    """Scores each passage against the query; higher is more relevant."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...

    def count_truncated(self, query: str, passages: Sequence[str]) -> int:
        """How many (query, passage) pairs exceed the model's input limit."""
        ...


class OverlapReranker:
    """Offline reranker: the share of the query's distinct word tokens found in the passage."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        wanted = set(tokenize_plain(query))
        if not wanted:
            return [0.0 for _ in passages]
        return [len(wanted & set(tokenize_plain(p))) / len(wanted) for p in passages]

    def count_truncated(self, query: str, passages: Sequence[str]) -> int:
        return 0


def rerank(
    query: str,
    hits: Sequence[RetrievalHit],
    chunks: Mapping[str, Chunk],
    reranker: Reranker,
    top_n: int,
) -> list[RetrievalHit]:
    """Re-order the first `top_n` hits by cross-encoder score; the rest are dropped."""
    candidates = list(hits[:top_n])
    if not candidates:
        return []
    scores = reranker.score(query, [chunks[h.chunk_id].text for h in candidates])
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [
        RetrievalHit(chunk_id=candidates[i].chunk_id, score=scores[i], rank=rank)
        for rank, i in enumerate(order, start=1)
    ]
