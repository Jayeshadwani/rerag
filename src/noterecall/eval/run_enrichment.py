"""The M5 enrichment runner. Implements ADR-010 (decisions) for ADR-003/ADR-007 (clean, blurb).

`run_enrichment_dev` runs the header, blurb and clean ablation of `enrichment.py` on the dev
split over the M4 pipeline and returns the contents of `results/enrichment_ablation_dev.json`:
per-query scores keyed by query id (never text), intervals and decisions, the LLM usage and cost
read from the response cache (reasoning and cached tokens shown separately), the clean report
(how many cells were changed or rejected, by reason), prompt versions, and the config snapshot.
`run_enrichment_test` evaluates the decided variant once on test, next to the M4 pipeline in the
same run; those comparisons are reported, never decided on.
"""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from noterecall.config import DecidedConfig, PriceConfig, Settings
from noterecall.eval.ablation import Rule, RunKey, compare
from noterecall.eval.enrichment import EnrichmentAblation, run_enrichment_ablation
from noterecall.eval.gold import GoldSet, Split
from noterecall.eval.run_retrieval import (
    SEED,
    Resources,
    RunContext,
    RunError,
    RunMeta,
    TestResults,
    make_evaluator,
    make_meta,
    score_params,
    select_queries,
)
from noterecall.ingest.llm_clean import CleanReport
from noterecall.llm.cache import ResponseCache
from noterecall.llm.cost import cost_usd
from noterecall.retrieval.pipeline import Variant


class UsageRow(BaseModel):
    """Token use and cost of every stored answer of one prompt and model."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    model: str
    calls: int
    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float


class LlmSummary(BaseModel):
    """What the LLM steps cost, from the response cache (the spend ledger)."""

    model_config = ConfigDict(frozen=True)

    rows: tuple[UsageRow, ...]
    total_cost_usd: float
    max_usd: float
    note: str = (
        "Answers that were paid for but unusable are not stored, so this is a lower bound by "
        "the cost of those calls."
    )


def llm_summary(
    cache: ResponseCache, prices: Mapping[str, PriceConfig], max_usd: float
) -> LlmSummary:
    """Usage and cost per (prompt, model) over everything in the cache."""
    rows = tuple(
        UsageRow(
            prompt=r.prompt_name,
            model=r.model,
            calls=r.calls,
            input_tokens=r.usage.input_tokens,
            cached_tokens=r.usage.cached_tokens,
            cache_write_tokens=r.usage.cache_write_tokens,
            output_tokens=r.usage.output_tokens,
            reasoning_tokens=r.usage.reasoning_tokens,
            cost_usd=cost_usd(r.usage, prices[r.model]),
        )
        for r in cache.spend()
    )
    return LlmSummary(rows=rows, total_cost_usd=sum(r.cost_usd for r in rows), max_usd=max_usd)


class VariantTruncation(BaseModel):
    """Chunks whose indexed text is cut at the dense model's input limit, for one variant."""

    model_config = ConfigDict(frozen=True)

    variant: Variant
    chunks: int
    truncated: int


class Extras(BaseModel):
    """What the runner records that comes from the enrichment steps, not from retrieval."""

    model_config = ConfigDict(frozen=True)

    llm: LlmSummary
    clean: CleanReport | None
    prompt_versions: dict[str, str]


class EnrichmentDevResults(BaseModel):
    """The contents of `results/enrichment_ablation_dev.json`."""

    model_config = ConfigDict(frozen=True)

    meta: RunMeta
    ablation: EnrichmentAblation
    truncation: tuple[VariantTruncation, ...]
    extras: Extras
    decided: DecidedConfig


def variant_truncation(
    resources: Resources, ablation: EnrichmentAblation, model: str
) -> tuple[VariantTruncation, ...]:
    """For each distinct variant that was run, how many indexed texts the dense model cuts."""
    service = resources.services[model]
    rows = []
    for config in ablation.configs:
        key = config.key
        texts = resources.indexed_texts(key)
        chunks = resources.chunks(key.cap, key.chunker, key.variant.clean)
        indexed = [texts[c.id] if texts is not None else c.text for c in chunks]
        rows.append(
            VariantTruncation(
                variant=key.variant, chunks=len(chunks), truncated=service.count_truncated(indexed)
            )
        )
    return tuple(rows)


def _decided_m4(settings: Settings) -> DecidedConfig:
    decided = settings.retrieval.decided
    if decided is None:
        raise RunError("config has no [retrieval.decided]; the M4 decisions are the base of M5")
    return decided


def run_enrichment_dev(
    settings: Settings, gold: GoldSet, resources: Resources, context: RunContext, extras: Extras
) -> EnrichmentDevResults:
    """Run header, blurb and clean on the dev split over the M4 pipeline."""
    base = _decided_m4(settings)
    queries = select_queries(gold, Split.DEV)
    evaluate = make_evaluator(resources, queries, score_params(settings))
    ablation = run_enrichment_ablation(evaluate, base.chunk_cap, base.pipeline, SEED)
    model = base.pipeline.dense_model
    truncation = variant_truncation(resources, ablation, model) if model else ()
    meta = make_meta(settings, Split.DEV, queries, context)
    return EnrichmentDevResults(
        meta=meta.model_copy(update={"prompt_versions": extras.prompt_versions}),
        ablation=ablation,
        truncation=truncation,
        extras=extras,
        decided=DecidedConfig(
            chunk_cap=base.chunk_cap, pipeline=base.pipeline, variant=ablation.variant
        ),
    )


def reference_variants(variant: Variant) -> list[Variant]:
    """What the decided variant is shown against on test: the M4 pipeline (no enrichment) and,
    when different, the header alone."""
    refs = [Variant()]
    header_only = Variant(header=True)
    if variant not in (Variant(), header_only):
        refs.append(header_only)
    return refs


def run_enrichment_test(
    settings: Settings,
    gold: GoldSet,
    resources: Resources,
    context: RunContext,
) -> TestResults:
    """Evaluate the decided variant once on the test split, next to the M4 pipeline."""
    decided = _decided_m4(settings)
    refs = select_queries(gold, Split.TEST)
    evaluate = make_evaluator(resources, refs, score_params(settings))
    cap, spec = decided.chunk_cap, decided.pipeline
    chosen = evaluate(RunKey(cap=cap, chunker="structural", spec=spec, variant=decided.variant))
    references = tuple(
        evaluate(RunKey(cap=cap, chunker="structural", spec=spec, variant=v))
        for v in reference_variants(decided.variant)
    )
    rule = Rule(stage="test (report only)", metric="recall@10", free=True)
    comparisons = tuple(
        compare(rule, chosen, ref, SEED).model_copy(update={"report_only": True})
        for ref in references
    )
    return TestResults(
        meta=make_meta(settings, Split.TEST, refs, context),
        milestone="M5",
        decided=decided,
        chosen=chosen,
        references=references,
        comparisons=comparisons,
    )
