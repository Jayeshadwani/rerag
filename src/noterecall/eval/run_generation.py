"""The M6 evaluation. Implements ADR-011 / proposal 006 (abstention) with ADR-009/010 (splits).

`run_calibration` computes the two stage-1 signals of every **dev** query (answerable and
unanswerable) with the decided pipeline, applies the pre-registered rule of
`generate/abstain.py`, and returns `results/abstention_calibration_dev.json`: per-query signals
keyed by query id (never text), the full threshold grid and the two stage-1 choices. It makes no
LLM call. `run_generation` asks every query of a split with the shipped thresholds and reports
the three rows of proposal 006 (cosine only, cosine + BM25, full two-stage), the first-attempt
citation-validity counts and the token usage and cost. The primary metric is the false-refusal
rate; unanswerable recall is given as raw counts; precision and F1 are for reference.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from noterecall.config import DecidedThresholds, Settings
from noterecall.eval.ablation import RunKey
from noterecall.eval.gold import GoldSet, QueryType, Split, relevant_keys
from noterecall.eval.metrics import is_relevant
from noterecall.eval.run_retrieval import (
    QueryRef,
    Resources,
    RunContext,
    RunError,
    RunMeta,
    make_meta,
)
from noterecall.generate.abstain import (
    Calibration,
    CalibrationPoint,
    RefusalMetrics,
    Thresholds,
    calibrate,
    refusal_metrics,
)
from noterecall.generate.answer import FirstAttempt, Generator, Outcome
from noterecall.generate.ask import Asker
from noterecall.llm.client import Usage
from noterecall.llm.cost import cost_usd
from noterecall.models import Chunk
from noterecall.retrieval.pipeline import AbstentionSignals, RetrievalPipeline


def select_all(gold: GoldSet, split: Split) -> list[QueryRef]:
    """Every query of a split, unanswerable ones included (they have no relevant cells)."""
    if not gold.frozen:
        raise RunError("the gold set is not frozen; the dev/test split is not assigned")
    refs = [
        QueryRef(q.id, q.text, q.type, relevant_keys(q)) for q in gold.queries if q.split is split
    ]
    if not any(r.type is QueryType.UNANSWERABLE for r in refs) or all(
        r.type is QueryType.UNANSWERABLE for r in refs
    ):
        raise RunError(f"the {split.value} split needs both answerable and unanswerable queries")
    return refs


def decided_pipeline(settings: Settings, resources: Resources) -> tuple[RetrievalPipeline, int]:
    """The M4/M5 decided pipeline (hybrid, header as decided) and its chunk cap."""
    decided = settings.retrieval.decided
    if decided is None:
        raise RunError("config has no [retrieval.decided]; M6 builds on the M4/M5 decisions")
    key = RunKey(cap=decided.chunk_cap, chunker="structural", spec=decided.pipeline,
                 variant=decided.variant)  # fmt: skip
    return resources.pipeline(key), decided.chunk_cap


class QueryPoint(BaseModel):
    """One dev query's stage-1 signals. Ids and numbers only."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    type: QueryType
    unanswerable: bool
    top_cosine: float
    bm25_norm: float


class CalibrationResults(BaseModel):
    """The contents of `results/abstention_calibration_dev.json`."""

    model_config = ConfigDict(frozen=True)

    meta: RunMeta
    points: tuple[QueryPoint, ...]
    calibration: Calibration


def run_calibration(
    settings: Settings, gold: GoldSet, resources: Resources, context: RunContext
) -> CalibrationResults:
    """Pre-registered stage-1 calibration on the dev split (no LLM)."""
    queries = select_all(gold, Split.DEV)
    pipeline, _cap = decided_pipeline(settings, resources)
    pipeline.retrieve_with_signals(queries[0].text)  # warm-up: models loaded, not recorded
    points = []
    for q in queries:
        _hits, signals = pipeline.retrieve_with_signals(q.text)
        points.append(
            QueryPoint(
                query_id=q.id,
                type=q.type,
                unanswerable=q.type is QueryType.UNANSWERABLE,
                top_cosine=signals.top_cosine,
                bm25_norm=signals.bm25_norm,
            )
        )
    cal_points = [
        CalibrationPoint(
            p.query_id, p.unanswerable, AbstentionSignals(top_cosine=p.top_cosine,
                                                          bm25_norm=p.bm25_norm)
        )
        for p in points
    ]  # fmt: skip
    calibration = calibrate(cal_points, settings.generation.max_false_refusal_pct)
    return CalibrationResults(
        meta=make_meta(settings, Split.DEV, queries, context),
        points=tuple(points),
        calibration=calibration,
    )


class QueryOutcome(BaseModel):
    """What happened to one query. No text, no answer, no citation text."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    type: QueryType
    unanswerable: bool
    top_cosine: float
    bm25_norm: float
    outcome: Outcome
    attempts: int
    first_attempt: FirstAttempt | None
    claims: int
    cites_relevant_chunk: bool | None
    usage: Usage
    cost_usd: float


class Validity(BaseModel):
    """First-attempt citation validity over the queries that reached the model."""

    model_config = ConfigDict(frozen=True)

    queries_to_model: int
    first_attempt_unparsable: int
    first_attempt_valid: int
    retried: int
    refused_invalid: int
    refused_by_model: int
    answered: int
    claims_first_attempt: int
    uncited_claims_first_attempt: int
    unknown_citations_first_attempt: int
    answered_answerable: int
    answered_citing_a_relevant_chunk: int


class Row(BaseModel):
    """One of the three reported rows."""

    model_config = ConfigDict(frozen=True)

    name: str
    metrics: RefusalMetrics


class GenerationResults(BaseModel):
    """The contents of `results/generation_<split>.json`."""

    model_config = ConfigDict(frozen=True)

    meta: RunMeta
    thresholds: DecidedThresholds
    cosine_only_threshold: float
    rows: tuple[Row, ...]
    validity: Validity
    queries: tuple[QueryOutcome, ...]
    total_usage: Usage
    total_cost_usd: float


def _outcome(asker: Asker, q: QueryRef, chunks: dict[str, Chunk]) -> QueryOutcome:
    result = asker.ask(q.text)
    r = result.response
    cites: bool | None = None
    if r.outcome is Outcome.ANSWERED and q.relevant:
        cited = {cid for claim in r.claims for cid in claim.chunk_ids}
        cites = any(is_relevant(chunks[cid], q.relevant) for cid in cited)
    return QueryOutcome(
        query_id=q.id,
        type=q.type,
        unanswerable=q.type is QueryType.UNANSWERABLE,
        top_cosine=result.signals.top_cosine,
        bm25_norm=result.signals.bm25_norm,
        outcome=r.outcome,
        attempts=r.attempts,
        first_attempt=r.first_attempt,
        claims=len(r.claims),
        cites_relevant_chunk=cites,
        usage=r.usage,
        cost_usd=0.0,
    )


def summarize_validity(outcomes: Sequence[QueryOutcome]) -> Validity:
    """Counts over the queries that reached the model."""
    called = [o for o in outcomes if o.attempts > 0 and o.first_attempt is not None]
    firsts = [o.first_attempt for o in called if o.first_attempt is not None]
    answered = [o for o in outcomes if o.outcome is Outcome.ANSWERED]
    answerable_answered = [o for o in answered if not o.unanswerable]
    return Validity(
        queries_to_model=len(called),
        first_attempt_unparsable=sum(1 for f in firsts if not f.parsed),
        first_attempt_valid=sum(1 for f in firsts if f.valid),
        retried=sum(1 for o in called if o.attempts == 2),
        refused_invalid=sum(1 for o in outcomes if o.outcome is Outcome.REFUSED_INVALID),
        refused_by_model=sum(1 for o in outcomes if o.outcome is Outcome.REFUSED_MODEL),
        answered=len(answered),
        claims_first_attempt=sum(f.claims for f in firsts),
        uncited_claims_first_attempt=sum(f.uncited_claims for f in firsts),
        unknown_citations_first_attempt=sum(f.unknown_citations for f in firsts),
        answered_answerable=len(answerable_answered),
        answered_citing_a_relevant_chunk=sum(
            1 for o in answerable_answered if o.cites_relevant_chunk
        ),
    )


def run_generation(  # noqa: PLR0913, PLR0917 - settings, data, models, provenance and split
    settings: Settings,
    gold: GoldSet,
    resources: Resources,
    context: RunContext,
    generator: Generator,
    split: Split,
) -> GenerationResults:
    """Ask every query of the split with the shipped thresholds and report the three rows."""
    gen = settings.generation
    if gen.decided is None or gen.cosine_only is None:
        raise RunError("config has no [generation.decided]; run and commit the dev calibration")
    queries = select_all(gold, split)
    pipeline, cap = decided_pipeline(settings, resources)
    chunks = {c.id: c for c in resources.chunks(cap, "structural")}
    shipped = Thresholds(cosine=gen.decided.cosine, bm25=gen.decided.bm25)
    asker = Asker(pipeline, chunks, shipped, generator, gen.max_chunks)
    price = settings.llm.prices[settings.llm.model]
    outcomes: list[QueryOutcome] = []
    for q in queries:
        o = _outcome(asker, q, chunks)
        outcomes.append(o.model_copy(update={"cost_usd": cost_usd(o.usage, price)}))
    rows = (
        Row(name="cosine only", metrics=refusal_metrics(
            (o.unanswerable, o.top_cosine < gen.cosine_only) for o in outcomes)),
        Row(name="cosine + BM25 (stage 1)", metrics=refusal_metrics(
            (o.unanswerable, o.outcome is Outcome.REFUSED_STAGE1) for o in outcomes)),
        Row(name="two-stage (ships)", metrics=refusal_metrics(
            (o.unanswerable, o.outcome is not Outcome.ANSWERED) for o in outcomes)),
    )  # fmt: skip
    usage = Usage()
    for o in outcomes:
        usage = usage.plus(o.usage)
    meta = make_meta(settings, split, queries, context)
    return GenerationResults(
        meta=meta.model_copy(update={"prompt_versions": {"answer": generator.prompt_version}}),
        thresholds=gen.decided,
        cosine_only_threshold=gen.cosine_only,
        rows=rows,
        validity=summarize_validity(outcomes),
        queries=tuple(outcomes),
        total_usage=usage,
        total_cost_usd=sum(o.cost_usd for o in outcomes),
    )
