"""Shared helpers for the retrieval and enrichment tests: a synthetic gold set over the fixture
notebooks (in memory only) and offline resources (hash embedder, overlap reranker)."""

from noterecall.chunking.tokens import Tokenizer
from noterecall.config import DecidedConfig, DecidedThresholds, Settings
from noterecall.eval.gold import (
    CellRef,
    GoldQuery,
    GoldSet,
    QueryType,
    freeze_gold,
)
from noterecall.eval.run_retrieval import Resources, RunContext
from noterecall.index.embeddings import EmbeddingService, HashEmbedder
from noterecall.ingest.stats import GitState
from noterecall.models import Cell
from noterecall.retrieval.pipeline import Depths, PipelineSpec, Variant
from noterecall.retrieval.rerank import OverlapReranker

MARKER = "ZEBRAMARK"
GIT = GitState(sha="c" * 40, dirty=False)
CONTEXT = RunContext(git=GIT, versions={"numpy": "0"})
# (query text, notebook, cell index): each query's answer is one cell of the fixtures
TARGETS = [
    ("groupby mean of each key", "01_no_headings.ipynb", 3),
    ("cross_val_score k-fold estimates", "02_deep_headings.ipynb", 5),
    ("train_test_split import", "02_deep_headings.ipynb", 4),
    ("groupby drops rows whose key is NaN", "01_no_headings.ipynb", 5),
]


def make_gold(notebooks: list[tuple[str, list[Cell]]], frozen: bool = True) -> GoldSet:
    hashes = {(p, c.index): c.raw_hash for p, cells in notebooks for c in cells}
    queries = []
    for n in range(24):
        text, path, index = TARGETS[n % len(TARGETS)]
        if n % 6 == 5:
            queries.append(
                GoldQuery(id=f"q{n:02d}", text=f"{MARKER} nothing", type=QueryType.UNANSWERABLE)
            )
            continue
        qtype = [QueryType.CONCEPT, QueryType.CODE_IDENTIFIER, QueryType.HOW_DID_I][n % 3]
        ref = CellRef(notebook_path=path, raw_hash=hashes[(path, index)])
        queries.append(
            GoldQuery(id=f"q{n:02d}", text=f"{text} {MARKER}", type=qtype, relevant=(ref,))
        )
    gold = GoldSet(queries=tuple(queries))
    return freeze_gold(gold) if frozen else gold


def make_resources(
    settings: Settings, notebooks: list[tuple[str, list[Cell]]], tok: Tokenizer
) -> Resources:
    services = {
        spec.id: EmbeddingService(HashEmbedder(), spec) for spec in settings.embeddings.candidates
    }
    depths = Depths(
        candidate=settings.retrieval.candidate_depth, fused=settings.retrieval.fused_depth
    )
    return Resources(notebooks, tok, services, OverlapReranker(), depths)


def decided_settings(
    settings: Settings,
    thresholds: tuple[float, float] | None = None,
    variant: Variant | None = None,
) -> Settings:
    """Settings with the M4/M5 decisions (hybrid at cap 64, small model) and, if given, the
    shipped stage-1 thresholds (cosine, BM25) and the cosine-only reference threshold."""
    decided = DecidedConfig(
        chunk_cap=64,
        pipeline=PipelineSpec(lexical="code", dense_model="small", rrf_k=60),
        variant=variant or Variant(),
    )
    retrieval = settings.retrieval.model_copy(update={"decided": decided})
    generation = settings.generation
    if thresholds is not None:
        generation = generation.model_copy(
            update={
                "decided": DecidedThresholds(cosine=thresholds[0], bm25=thresholds[1]),
                "cosine_only": thresholds[0],
            }
        )
    return settings.model_copy(update={"retrieval": retrieval, "generation": generation})
