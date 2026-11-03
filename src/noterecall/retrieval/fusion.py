"""Reciprocal rank fusion. Implements ADR-005 (RRF, Cormack, Clarke and Buettcher, SIGIR 2009).

Each ranked list contributes `1 / (k + rank)` to every chunk it contains, with ranks starting at
1; a chunk's fused score is the sum over lists. Only ranks are used, so score scales of the
retrievers never matter. Ties in the fused score are broken by chunk id, which makes the result
independent of the order the lists are given in. The terms are added with `math.fsum`, which is
exactly rounded, so the float result does not depend on the order either (a plain running sum
differs in the last bit, which could flip a tie).
"""

import math
from collections.abc import Sequence

from noterecall.models import RetrievalHit


class FusionError(Exception):
    """Raised when fusion is asked for something undefined."""


def rrf(rankings: Sequence[Sequence[str]], k: int) -> list[RetrievalHit]:
    """Fuse ranked lists of chunk ids into one ranked list of hits."""
    if k < 1:
        raise FusionError(f"k must be at least 1, got {k}")
    terms: dict[str, list[float]] = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise FusionError("a ranking lists the same chunk more than once")
        for rank, chunk_id in enumerate(ranking, start=1):
            terms.setdefault(chunk_id, []).append(1.0 / (k + rank))
    scores = {chunk_id: math.fsum(parts) for chunk_id, parts in terms.items()}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        RetrievalHit(chunk_id=chunk_id, score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]
