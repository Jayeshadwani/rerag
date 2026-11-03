"""Exact dense search, the reranker stand-in and the rerank step."""

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest

from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec
from noterecall.models import Chunk, RetrievalHit
from noterecall.retrieval.dense import DenseError, DenseIndex
from noterecall.retrieval.rerank import OverlapReranker, rerank

SPEC = EmbeddingSpec(id="m", revision="r", query_prefix="", passage_prefix="")


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid, notebook_path="n", heading_path=(), pieces=(), text=text, tokens=1, cell_hashes=()
    )


class Fixed:
    """Embedder with hand-set 2-d unit vectors keyed by text."""

    VECTORS: ClassVar[dict[str, list[float]]] = {
        "east": [1.0, 0.0],
        "north": [0.0, 1.0],
        "north-east": [0.6, 0.8],
        "query-ne": [0.8, 0.6],
        "query-e": [1.0, 0.0],
    }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self.VECTORS[t] for t in texts], dtype=np.float32)

    def count_truncated(self, texts: Sequence[str]) -> int:
        return 0


def test_cosine_ranking_matches_hand_computed_scores() -> None:
    chunks = [chunk("a", "east"), chunk("b", "north"), chunk("c", "north-east")]
    index = DenseIndex.build(chunks, EmbeddingService(Fixed(), SPEC))
    hits = index.search("query-ne", 3)
    # query (0.8, 0.6): a = 0.8, b = 0.6, c = 0.6 * 0.8 + 0.8 * 0.6 = 0.96
    assert [h.chunk_id for h in hits] == ["c", "a", "b"]
    assert [h.score for h in hits] == pytest.approx([0.96, 0.8, 0.6])
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.chunk_id for h in index.search("query-ne", 1)] == ["c"]


def test_equal_scores_keep_chunk_order() -> None:
    chunks = [chunk("z", "east"), chunk("a", "east")]
    hits = DenseIndex.build(chunks, EmbeddingService(Fixed(), SPEC)).search("query-e", 2)
    assert [h.chunk_id for h in hits] == ["z", "a"]


def test_dense_errors() -> None:
    service = EmbeddingService(Fixed(), SPEC)
    with pytest.raises(DenseError, match="zero chunks"):
        DenseIndex.build([], service)
    with pytest.raises(DenseError, match="k must be at least 1"):
        DenseIndex.build([chunk("a", "east")], service).search("query-e", 0)


def test_overlap_reranker_scores_query_token_coverage() -> None:
    scores = OverlapReranker().score(
        "train test split", ["the train test split helper", "only train here", "nothing"]
    )
    assert scores == pytest.approx([1.0, 1 / 3, 0.0])
    assert OverlapReranker().score("", ["x", "y"]) == [0.0, 0.0]
    assert OverlapReranker().count_truncated("q", ["p"]) == 0


def hits_of(*ids: str) -> list[RetrievalHit]:
    return [RetrievalHit(chunk_id=i, score=1.0 / n, rank=n) for n, i in enumerate(ids, start=1)]


def test_rerank_reorders_the_top_n_and_drops_the_rest() -> None:
    chunks = {
        "a": chunk("a", "irrelevant words"),
        "b": chunk("b", "train test split"),
        "c": chunk("c", "train only"),
        "d": chunk("d", "train test split"),
    }
    out = rerank("train test split", hits_of("a", "b", "c", "d"), chunks, OverlapReranker(), 3)
    # candidates a, b, c score 0, 1, 1/3; d is outside the top 3 and is dropped
    assert [h.chunk_id for h in out] == ["b", "c", "a"]
    assert [h.rank for h in out] == [1, 2, 3]
    assert [h.score for h in out] == pytest.approx([1.0, 1 / 3, 0.0])


def test_rerank_ties_keep_the_fused_order_and_empty_input_is_empty() -> None:
    chunks = {"a": chunk("a", "same"), "b": chunk("b", "same")}
    out = rerank("same", hits_of("b", "a"), chunks, OverlapReranker(), 2)
    assert [h.chunk_id for h in out] == ["b", "a"]
    assert rerank("q", [], chunks, OverlapReranker(), 5) == []
