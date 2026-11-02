"""The staged pipeline: wiring, depths, fusion, reranking, timing and spec validation."""

import pytest

from noterecall.index.code_tokenizer import tokenize_plain
from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec, HashEmbedder
from noterecall.index.lexical import LexicalIndex
from noterecall.models import Chunk
from noterecall.retrieval.dense import DenseIndex
from noterecall.retrieval.pipeline import (
    Components,
    Depths,
    PipelineError,
    PipelineSpec,
    RetrievalPipeline,
)
from noterecall.retrieval.rerank import OverlapReranker

SPEC = EmbeddingSpec(id="m", revision="r", query_prefix="", passage_prefix="")
TEXTS = {
    "c0": "pandas groupby aggregate mean",
    "c1": "train test split for model evaluation",
    "c2": "numpy array reshape",
    "c3": "cross validation scores for the model",
    "c4": "plotting with matplotlib",
}
CHUNKS = {
    cid: Chunk(
        id=cid, notebook_path="n", heading_path=(), pieces=(), text=t, tokens=1, cell_hashes=()
    )
    for cid, t in TEXTS.items()
}
DEPTHS = Depths(candidate=4, fused=3)


def parts(lexical: bool = True, dense: bool = True, reranker: bool = False) -> Components:
    chunks = list(CHUNKS.values())
    return Components(
        lexical=LexicalIndex.build(chunks, tokenize_plain) if lexical else None,
        dense=DenseIndex.build(chunks, EmbeddingService(HashEmbedder(), SPEC)) if dense else None,
        reranker=OverlapReranker() if reranker else None,
    )


def test_lexical_only_returns_matching_chunks_up_to_the_fused_depth() -> None:
    pipeline = RetrievalPipeline(PipelineSpec(lexical="plain"), CHUNKS, DEPTHS, parts(dense=False))
    hits, times = pipeline.retrieve("model")
    assert [h.chunk_id for h in hits] == ["c1", "c3"]  # only chunks with the word
    assert times.lexical_ms >= 0.0
    assert times.dense_ms == 0.0
    assert times.rerank_ms == 0.0
    assert times.total_ms >= times.lexical_ms


def test_dense_only_returns_the_fused_depth_best_chunks() -> None:
    pipeline = RetrievalPipeline(
        PipelineSpec(dense_model="m"), CHUNKS, DEPTHS, parts(lexical=False)
    )
    hits, times = pipeline.retrieve("train test split model")
    assert len(hits) == 3
    assert hits[0].chunk_id == "c1"
    assert times.lexical_ms == 0.0
    assert times.dense_ms >= 0.0


def test_hybrid_fuses_both_lists_with_rrf() -> None:
    spec = PipelineSpec(lexical="plain", dense_model="m", rrf_k=60)
    pipeline = RetrievalPipeline(spec, CHUNKS, DEPTHS, parts())
    hits, times = pipeline.retrieve("train test split model")
    assert hits[0].chunk_id == "c1"  # first in both lists
    assert hits[0].score == pytest.approx(2 / 61)
    assert len(hits) == 3
    assert times.fusion_ms >= 0.0


def test_rerank_reorders_the_fused_candidates() -> None:
    spec = PipelineSpec(lexical="plain", dense_model="m", rrf_k=60, rerank=True)
    pipeline = RetrievalPipeline(spec, CHUNKS, DEPTHS, parts(reranker=True))
    hits, times = pipeline.retrieve("cross validation scores")
    assert hits[0].chunk_id == "c3"  # the overlap reranker puts the exact match first
    assert hits[0].score == pytest.approx(1.0)
    assert times.rerank_ms >= 0.0


def test_no_match_gives_no_hits_for_lexical_only() -> None:
    pipeline = RetrievalPipeline(PipelineSpec(lexical="plain"), CHUNKS, DEPTHS, parts(dense=False))
    assert pipeline.retrieve("zzz")[0] == []


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="at least one retriever"):
        PipelineSpec()
    with pytest.raises(ValueError, match="rrf_k must be set"):
        PipelineSpec(lexical="code", dense_model="m")
    with pytest.raises(ValueError, match="rrf_k must be set"):
        PipelineSpec(lexical="code", rrf_k=60)
    assert PipelineSpec(lexical="code", dense_model="m", rrf_k=60).rerank is False


def test_components_must_match_the_spec() -> None:
    with pytest.raises(PipelineError, match="lexical index"):
        RetrievalPipeline(PipelineSpec(lexical="plain"), CHUNKS, DEPTHS, Components())
    with pytest.raises(PipelineError, match="lexical index"):
        RetrievalPipeline(PipelineSpec(dense_model="m"), CHUNKS, DEPTHS, parts(dense=True))
    with pytest.raises(PipelineError, match="dense index"):
        RetrievalPipeline(PipelineSpec(dense_model="m"), CHUNKS, DEPTHS, Components())
    with pytest.raises(PipelineError, match="dense index"):
        RetrievalPipeline(PipelineSpec(lexical="plain"), CHUNKS, DEPTHS, parts())
    with pytest.raises(PipelineError, match="no reranker"):
        RetrievalPipeline(
            PipelineSpec(lexical="plain", rerank=True), CHUNKS, DEPTHS, parts(dense=False)
        )
