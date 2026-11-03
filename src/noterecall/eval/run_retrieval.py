"""The retrieval ablation runner. Implements ADR-010 (decisions) and ADR-009 (frozen split).

`run_dev` builds the corpus at each chunk cap, runs the staged ablation of `ablation.py` on the
dev split, and returns everything needed for `results/retrieval_ablation_dev.json`: per-query
scores keyed by query id (never query text), intervals and decisions, latency per stage, config
snapshot, model revisions, library versions and the git SHA. `run_test` evaluates only the
decided pipeline and the BM25 baseline on the test split, once. Models and caches are passed
in, so the whole runner is tested offline with the hash embedder and the overlap reranker.
"""

import os
import platform
from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import structlog
from pydantic import BaseModel, ConfigDict

from noterecall.chunking.fixed import fixed_chunks
from noterecall.chunking.structural import chunk_notebook
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import DecidedConfig, Settings
from noterecall.enrich.headers import index_text
from noterecall.eval.ablation import (
    Ablation,
    AblationParams,
    Chunker,
    Comparison,
    ConfigResult,
    Evaluator,
    Rule,
    RunKey,
    compare,
    describe,
    run_ablation,
    summarize_latency,
)
from noterecall.eval.coverage import CoverageIndex
from noterecall.eval.gold import CellKey, GoldSet, QueryType, Split, relevant_keys
from noterecall.eval.metrics import score_query
from noterecall.eval.stats import BOOTSTRAP_RESAMPLES, CONFIDENCE
from noterecall.index.code_tokenizer import tokenize_code, tokenize_plain
from noterecall.index.embeddings import EmbeddingService
from noterecall.index.lexical import LexicalIndex
from noterecall.ingest.stats import GitState
from noterecall.models import Cell, Chunk
from noterecall.retrieval.dense import DenseIndex
from noterecall.retrieval.pipeline import (
    Components,
    Depths,
    PipelineSpec,
    RetrievalPipeline,
    Variant,
)
from noterecall.retrieval.rerank import Reranker

log = structlog.get_logger()

SEED = 20261029
KS = (5, 10)
Notebooks = Sequence[tuple[str, list[Cell]]]


class RunError(Exception):
    """Raised when the runner cannot proceed (empty split, duplicate chunk ids, no models)."""


class QueryRef(NamedTuple):
    """An answerable gold query: text stays in memory, only the id leaves."""

    id: str
    text: str
    type: QueryType
    relevant: frozenset[CellKey]


def select_queries(gold: GoldSet, split: Split) -> list[QueryRef]:
    """Answerable queries of one split (unanswerable ones have no retrieval score)."""
    if not gold.frozen:
        raise RunError("the gold set is not frozen; the dev/test split is not assigned")
    refs = [
        QueryRef(q.id, q.text, q.type, relevant_keys(q))
        for q in gold.queries
        if q.split is split and q.type is not QueryType.UNANSWERABLE
    ]
    if not refs:
        raise RunError(f"the {split.value} split has no answerable queries")
    return refs


def build_chunks(
    notebooks: Notebooks, cap: int, chunker: Chunker, tokenizer: Tokenizer
) -> list[Chunk]:
    """Chunk every notebook with the structural chunker or the fixed-token baseline."""
    chunks: list[Chunk] = []
    for path, cells in notebooks:
        if chunker == "structural":
            chunks.extend(chunk_notebook(path, cells, cap, tokenizer))
        else:
            chunks.extend(fixed_chunks(path, cells, cap, tokenizer))
    if len({c.id for c in chunks}) != len(chunks):
        raise RunError("two chunks share an id; chunk ids must be unique across the corpus")
    return chunks


class Resources:
    """Builds corpora, indexes and pipelines on demand and keeps them (each is expensive).

    A run key's variant decides the text that is indexed (ADR-007): the chunk text, or the
    deterministic header followed by it. Cited and displayed text is always the chunk's own.
    """

    def __init__(
        self,
        notebooks: Notebooks,
        tokenizer: Tokenizer,
        services: Mapping[str, EmbeddingService],
        reranker: Reranker,
        depths: Depths,
    ) -> None:
        self._notebooks = notebooks
        self._tokenizer = tokenizer
        self.services = services
        self.reranker = reranker
        self._depths = depths
        self._chunks: dict[tuple[int, Chunker], list[Chunk]] = {}
        self._lexical: dict[tuple[int, Chunker, str, Variant], LexicalIndex] = {}
        self._dense: dict[tuple[int, Chunker, str, Variant], DenseIndex] = {}
        self._coverage: CoverageIndex | None = None

    def chunks(self, cap: int, chunker: Chunker) -> list[Chunk]:
        if (cap, chunker) not in self._chunks:
            self._chunks[(cap, chunker)] = build_chunks(
                self._notebooks, cap, chunker, self._tokenizer
            )
        return self._chunks[(cap, chunker)]

    def indexed_texts(self, key: RunKey) -> dict[str, str] | None:
        """The text each chunk is indexed under, or None for the plain chunk text."""
        if not key.variant.header:
            return None
        return {c.id: index_text(c, True) for c in self.chunks(key.cap, key.chunker)}

    def pipeline(self, key: RunKey) -> RetrievalPipeline:
        chunks = self.chunks(key.cap, key.chunker)
        spec = key.spec
        texts = self.indexed_texts(key)
        lexical = dense = None
        if spec.lexical is not None:
            lk = (key.cap, key.chunker, spec.lexical, key.variant)
            if lk not in self._lexical:
                tokenize = tokenize_code if spec.lexical == "code" else tokenize_plain
                self._lexical[lk] = LexicalIndex.build(chunks, tokenize, texts)
            lexical = self._lexical[lk]
        if spec.dense_model is not None:
            dk = (key.cap, key.chunker, spec.dense_model, key.variant)
            if dk not in self._dense:
                service = self._service(spec.dense_model)
                self._dense[dk] = DenseIndex.build(chunks, service, texts)
            dense = self._dense[dk]
        parts = Components(lexical=lexical, dense=dense, reranker=self.reranker)
        return RetrievalPipeline(spec, {c.id: c for c in chunks}, self._depths, parts)

    def coverage(self) -> CoverageIndex:
        """Token counts of every cell, for the secondary token-coverage metric."""
        if self._coverage is None:
            self._coverage = CoverageIndex(self._notebooks, self._tokenizer)
        return self._coverage

    def _service(self, model: str) -> EmbeddingService:
        if model not in self.services:
            raise RunError(f"no embedding model {model!r} was provided")
        return self.services[model]


class ScoreParams(BaseModel):
    """Which metrics to compute per query."""

    model_config = ConfigDict(frozen=True)

    ks: tuple[int, ...]
    budgets: tuple[int, ...]


def make_evaluator(
    resources: Resources, queries: Sequence[QueryRef], params: ScoreParams
) -> Evaluator:
    """An `Evaluator` that runs a configuration over the queries and scores it."""

    def evaluate(key: RunKey) -> ConfigResult:
        pipeline = resources.pipeline(key)
        coverage = resources.coverage()
        chunks = resources.chunks(key.cap, key.chunker)
        by_id = {c.id: c for c in chunks}
        pipeline.retrieve(queries[0].text)  # warm-up: models loaded, caches filled, not timed
        per_query: dict[str, dict[str, float]] = {}
        times = []
        for query in queries:
            hits, stage_times = pipeline.retrieve(query.text)
            times.append(stage_times)
            ranked = [by_id[h.chunk_id] for h in hits]
            scores = score_query(
                ranked, query.relevant, chunks, ks=params.ks, budgets=params.budgets
            )
            for budget in params.budgets:
                scores[f"tokcov@tokens={budget}"] = coverage.at_budget(
                    ranked, query.relevant, budget
                )
            per_query[query.id] = scores
        metrics = next(iter(per_query.values())).keys()
        means = {m: sum(s[m] for s in per_query.values()) / len(per_query) for m in metrics}
        log.info("eval.config", name=describe(key), recall_at_10=round(means["recall@10"], 4))
        return ConfigResult(
            name=describe(key),
            key=key,
            means=means,
            per_query=per_query,
            latency=summarize_latency(times),
        )

    return evaluate


class TruncationRow(BaseModel):
    """How many chunks an embedding model cuts at its input limit (never silent)."""

    model_config = ConfigDict(frozen=True)

    model: str
    cap: int
    chunks: int
    truncated: int


class HardwareInfo(BaseModel):
    """Where the latencies were measured."""

    model_config = ConfigDict(frozen=True)

    platform: str
    machine: str
    python: str
    cpu_count: int | None


class RunContext(BaseModel):
    """The commit and library versions a run is attributed to."""

    model_config = ConfigDict(frozen=True)

    git: GitState
    versions: dict[str, str]


class RunMeta(BaseModel):
    """Provenance shared by every results file (CLAUDE.md section 4)."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1]
    split: Split
    git: GitState
    config: dict[str, object]
    tokenizer: str
    versions: dict[str, str]
    hardware: HardwareInfo
    prompt_versions: dict[str, str]
    bootstrap_resamples: int
    confidence: float
    seed: int
    queries: int
    query_types: dict[str, str]


class DevResults(BaseModel):
    """The contents of `results/retrieval_ablation_dev.json`."""

    model_config = ConfigDict(frozen=True)

    meta: RunMeta
    params: AblationParams
    ablation: Ablation
    truncation: tuple[TruncationRow, ...]
    sensitivity: tuple[ConfigResult, ...]
    decided: DecidedConfig
    notes: tuple[str, ...] = ()


class TestResults(BaseModel):
    """The contents of `results/retrieval_test_<milestone>.json`."""

    __test__ = False  # not a pytest class
    model_config = ConfigDict(frozen=True)

    meta: RunMeta
    milestone: str
    decided: DecidedConfig
    chosen: ConfigResult
    references: tuple[ConfigResult, ...]
    comparisons: tuple[Comparison, ...]


def hardware_info() -> HardwareInfo:
    """The machine the run happened on."""
    return HardwareInfo(
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
    )


def make_meta(
    settings: Settings, split: Split, queries: Sequence[QueryRef], context: RunContext
) -> RunMeta:
    """Provenance for a results file. Query types are keyed by id, never by text."""
    return RunMeta(
        schema_version=1,
        split=split,
        git=context.git,
        config=settings.model_dump(mode="json"),
        tokenizer=settings.tokenizer.encoding,
        versions=context.versions,
        hardware=hardware_info(),
        prompt_versions={},
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        confidence=CONFIDENCE,
        seed=SEED,
        queries=len(queries),
        query_types={q.id: q.type.value for q in queries},
    )


def ablation_params(settings: Settings) -> AblationParams:
    """The pre-registered ablation inputs, from config."""
    caps = settings.chunking.candidate_caps
    r = settings.retrieval
    return AblationParams(
        caps=caps,
        baseline_cap=sorted(caps)[len(caps) // 2],
        budget_metric=f"recall@tokens={r.budget_tokens}",
        secondary_metric=f"tokcov@tokens={r.budget_tokens}",
        embedding_models=tuple(c.id for c in settings.embeddings.candidates),
        rrf_default=r.rrf_default,
        rrf_challengers=r.rrf_challengers,
        latency_budget_ms=r.latency_budget_ms,
        seed=SEED,
    )


def score_params(settings: Settings) -> ScoreParams:
    """The metrics computed per query: Recall/MRR/nDCG at k, and recall at each token budget."""
    r = settings.retrieval
    return ScoreParams(ks=KS, budgets=(r.budget_tokens, *r.sensitivity_budgets))


def truncation_rows(
    resources: Resources, caps: Sequence[int], models: Sequence[str]
) -> tuple[TruncationRow, ...]:
    """Count, for each embedding model and structural cap, the chunks cut at the input limit."""
    rows = []
    for model in models:
        for cap in caps:
            chunks = resources.chunks(cap, "structural")
            service = resources.services[model]
            rows.append(
                TruncationRow(
                    model=model,
                    cap=cap,
                    chunks=len(chunks),
                    truncated=service.count_truncated([c.text for c in chunks]),
                )
            )
    return tuple(rows)


def limitation_notes(rows: Sequence[TruncationRow], decided: DecidedConfig) -> tuple[str, ...]:
    """Known limitations to print next to the results (the decided model's truncation)."""
    model = decided.pipeline.dense_model
    if model is None:
        return ()
    row = next((r for r in rows if r.model == model and r.cap == decided.chunk_cap), None)
    if row is None:
        raise RunError(f"no truncation count for {model} at cap {decided.chunk_cap}")
    share = 100 * row.truncated / row.chunks
    return (
        f"Known limitation: at chunk cap {row.cap}, {row.truncated} of {row.chunks} chunks "
        f"({share:.1f}%) are longer than the input limit of {model} and are cut, so dense "
        "retrieval does not see their tails. The chunker is unchanged.",
    )


def run_dev(
    settings: Settings, gold: GoldSet, resources: Resources, context: RunContext
) -> DevResults:
    """Run the full staged ablation on the dev split."""
    queries = select_queries(gold, Split.DEV)
    evaluate = make_evaluator(resources, queries, score_params(settings))
    params = ablation_params(settings)
    ablation = run_ablation(evaluate, params)
    sensitivity = tuple(
        evaluate(RunKey(cap=cap, chunker="structural", spec=ablation.pipeline))
        for cap in params.caps
    )
    truncation = truncation_rows(resources, params.caps, params.embedding_models)
    decided = DecidedConfig(chunk_cap=ablation.chunk_cap, pipeline=ablation.pipeline)
    return DevResults(
        meta=make_meta(settings, Split.DEV, queries, context),
        params=params,
        ablation=ablation,
        truncation=truncation,
        sensitivity=sensitivity,
        decided=decided,
        notes=limitation_notes(truncation, decided),
    )


def reference_specs(pipeline: PipelineSpec) -> list[PipelineSpec]:
    """The comparison rows for the test run: a hybrid is shown against each of its parts;
    anything else against the BM25 baseline."""
    if pipeline.lexical is not None and pipeline.dense_model is not None:
        return [
            PipelineSpec(dense_model=pipeline.dense_model),
            PipelineSpec(lexical=pipeline.lexical),
        ]
    baseline = PipelineSpec(lexical="code")
    return [] if pipeline == baseline else [baseline]


def comparison_rows(decided: DecidedConfig) -> list[tuple[PipelineSpec, Variant]]:
    """The comparison rows for the test run. With the header on, the first row is the same
    pipeline with the header off (the M4 pipeline); then the parts of a hybrid, as in M4."""
    rows = [(spec, decided.variant) for spec in reference_specs(decided.pipeline)]
    if decided.variant.header:
        rows.insert(0, (decided.pipeline, Variant()))
    return rows


def run_test(
    settings: Settings, gold: GoldSet, resources: Resources, context: RunContext, milestone: str
) -> TestResults:
    """Evaluate the decided pipeline and its comparison rows once on the test split.

    The comparisons are reported, not decided on: the test split is for reporting only.
    """
    decided = settings.retrieval.decided
    if decided is None:
        raise RunError(
            "config has no [retrieval.decided]; run the dev ablation and commit it first"
        )
    queries = select_queries(gold, Split.TEST)
    evaluate = make_evaluator(resources, queries, score_params(settings))
    cap = decided.chunk_cap
    chosen = evaluate(
        RunKey(cap=cap, chunker="structural", spec=decided.pipeline, variant=decided.variant)
    )
    references = tuple(
        evaluate(RunKey(cap=cap, chunker="structural", spec=spec, variant=ref_variant))
        for spec, ref_variant in comparison_rows(decided)
    )
    rule = Rule(stage="test (report only)", metric="recall@10", free=True)
    comparisons = tuple(
        compare(rule, chosen, ref, SEED).model_copy(update={"report_only": True})
        for ref in references
    )
    return TestResults(
        meta=make_meta(settings, Split.TEST, queries, context),
        milestone=milestone,
        decided=decided,
        chosen=chosen,
        references=references,
        comparisons=comparisons,
    )
