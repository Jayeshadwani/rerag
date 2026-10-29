"""Retrieval metrics at the cell level. Implements ADR-009 (labels) and ADR-010 (metrics).

Labels are cells `(notebook_path, raw_hash)`; retrieval returns chunks. A chunk covers the cells
named in its `cell_hashes` within its notebook, and is relevant if it covers any relevant cell.

- Recall@k: distinct relevant cells covered by the top-k chunks / number of relevant cells.
- MRR@k: 1 / rank of the first relevant chunk in the top k, else 0.
- nDCG@k: binary gain per chunk, log2 discount; the ideal ranking puts min(k, R) relevant chunks
  first, where R is the number of relevant chunks in the whole chunked corpus.
- Recall@token-budget B: take chunks in rank order while the running token total stays <= B and
  stop at the first chunk that does not fit (no skipping, no truncation); then cell recall.

Queries with no relevant cells (unanswerable) have no defined score: callers exclude them and
these functions raise `MetricsError` rather than return a made-up number.
"""

import math
from collections.abc import Collection, Sequence

from noterecall.eval.gold import CellKey
from noterecall.models import Chunk


class MetricsError(Exception):
    """Raised when a metric is asked for something it cannot define."""


def chunk_cells(chunk: Chunk) -> frozenset[CellKey]:
    """The (notebook_path, raw_hash) cells a chunk covers."""
    return frozenset((chunk.notebook_path, h) for h in chunk.cell_hashes)


def _check(ranked: Sequence[Chunk], relevant: Collection[CellKey]) -> None:
    if not relevant:
        raise MetricsError("a query with no relevant cells has no retrieval score")
    ids = [c.id for c in ranked]
    if len(ids) != len(set(ids)):
        raise MetricsError("a ranking lists the same chunk more than once")


def _check_k(k: int) -> None:
    if k < 1:
        raise MetricsError(f"k must be at least 1, got {k}")


def _covered(chunks: Sequence[Chunk], relevant: Collection[CellKey]) -> int:
    seen: set[CellKey] = set()
    for chunk in chunks:
        seen |= chunk_cells(chunk)
    return len(seen & set(relevant))


def is_relevant(chunk: Chunk, relevant: Collection[CellKey]) -> bool:
    """Whether the chunk covers at least one relevant cell."""
    return not chunk_cells(chunk).isdisjoint(relevant)


def recall_at_k(ranked: Sequence[Chunk], relevant: Collection[CellKey], k: int) -> float:
    """Fraction of relevant cells covered by the first k chunks."""
    _check_k(k)
    _check(ranked, relevant)
    return _covered(ranked[:k], relevant) / len(set(relevant))


def mrr_at_k(ranked: Sequence[Chunk], relevant: Collection[CellKey], k: int) -> float:
    """Reciprocal rank of the first relevant chunk within the first k, else 0."""
    _check_k(k)
    _check(ranked, relevant)
    for rank, chunk in enumerate(ranked[:k], start=1):
        if is_relevant(chunk, relevant):
            return 1.0 / rank
    return 0.0


def count_relevant_chunks(universe: Sequence[Chunk], relevant: Collection[CellKey]) -> int:
    """How many chunks of the chunked corpus are relevant (the nDCG ideal needs this)."""
    return sum(1 for chunk in universe if is_relevant(chunk, relevant))


def ndcg_at_k(
    ranked: Sequence[Chunk], relevant: Collection[CellKey], k: int, relevant_chunks: int
) -> float:
    """Binary-gain nDCG over the first k chunks; R comes from `count_relevant_chunks`."""
    _check_k(k)
    _check(ranked, relevant)
    if relevant_chunks < 1:
        raise MetricsError("no chunk of the corpus covers a relevant cell")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk in enumerate(ranked[:k], start=1)
        if is_relevant(chunk, relevant)
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, relevant_chunks) + 1))
    return dcg / ideal


def recall_at_token_budget(
    ranked: Sequence[Chunk], relevant: Collection[CellKey], budget: int
) -> float:
    """Cell recall of the longest rank-order prefix whose total tokens are within the budget."""
    if budget < 1:
        raise MetricsError(f"budget must be at least 1 token, got {budget}")
    _check(ranked, relevant)
    used = 0
    taken: list[Chunk] = []
    for chunk in ranked:
        if used + chunk.tokens > budget:
            break
        used += chunk.tokens
        taken.append(chunk)
    return _covered(taken, relevant) / len(set(relevant))


def score_query(
    ranked: Sequence[Chunk],
    relevant: Collection[CellKey],
    universe: Sequence[Chunk],
    ks: Sequence[int] = (5, 10),
    budgets: Sequence[int] = (),
) -> dict[str, float]:
    """All per-query scores, keyed like `recall@10`, `mrr@10`, `ndcg@10`, `recall@tokens=2000`."""
    relevant_chunks = count_relevant_chunks(universe, relevant)
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        scores[f"mrr@{k}"] = mrr_at_k(ranked, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k, relevant_chunks)
    for budget in budgets:
        scores[f"recall@tokens={budget}"] = recall_at_token_budget(ranked, relevant, budget)
    return scores
