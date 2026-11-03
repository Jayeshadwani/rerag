"""The M5 runner. Implements ADR-007 (header) with ADR-010 (reporting) and ADR-009 (dev split).

`run_header_dev` evaluates the M4 pipeline with and without the context header on the dev split and
returns the contents of `results/header_ablation_dev.json`: per-query scores keyed by query id
(never text), the interval, the count of chunks the dense model cuts with and without the header,
and the config snapshot. The header stays on whatever the row shows (ADR-007 is locked), so the
decided variant is always `header = true`. The LLM steps are not run (proposal 005).
"""

from pydantic import BaseModel, ConfigDict

from noterecall.config import DecidedConfig, Settings
from noterecall.eval.gold import GoldSet, Split
from noterecall.eval.header_ablation import HeaderAblation, run_header_ablation
from noterecall.eval.run_retrieval import (
    SEED,
    Resources,
    RunContext,
    RunError,
    RunMeta,
    make_evaluator,
    make_meta,
    score_params,
    select_queries,
)
from noterecall.retrieval.pipeline import Variant


class VariantTruncation(BaseModel):
    """Chunks whose indexed text is cut at the dense model's input limit, for one variant."""

    model_config = ConfigDict(frozen=True)

    variant: Variant
    chunks: int
    truncated: int


class HeaderDevResults(BaseModel):
    """The contents of `results/header_ablation_dev.json`."""

    model_config = ConfigDict(frozen=True)

    meta: RunMeta
    ablation: HeaderAblation
    truncation: tuple[VariantTruncation, ...]
    decided: DecidedConfig


def variant_truncation(
    resources: Resources, ablation: HeaderAblation, model: str
) -> tuple[VariantTruncation, ...]:
    """For each variant, how many indexed texts the dense model cuts at its input limit."""
    service = resources.services[model]
    rows = []
    for config in (ablation.without_header, ablation.with_header):
        key = config.key
        texts = resources.indexed_texts(key)
        chunks = resources.chunks(key.cap, key.chunker)
        indexed = [texts[c.id] if texts is not None else c.text for c in chunks]
        truncated = service.count_truncated(indexed)
        rows.append(VariantTruncation(variant=key.variant, chunks=len(chunks), truncated=truncated))
    return tuple(rows)


def run_header_dev(
    settings: Settings, gold: GoldSet, resources: Resources, context: RunContext
) -> HeaderDevResults:
    """Run header on vs off on the dev split over the M4 decided pipeline."""
    base = settings.retrieval.decided
    if base is None:
        raise RunError("config has no [retrieval.decided]; the M4 decisions are the base of M5")
    queries = select_queries(gold, Split.DEV)
    evaluate = make_evaluator(resources, queries, score_params(settings))
    ablation = run_header_ablation(evaluate, base.chunk_cap, base.pipeline, SEED)
    model = base.pipeline.dense_model
    truncation = variant_truncation(resources, ablation, model) if model else ()
    return HeaderDevResults(
        meta=make_meta(settings, Split.DEV, queries, context),
        ablation=ablation,
        truncation=truncation,
        decided=DecidedConfig(
            chunk_cap=base.chunk_cap, pipeline=base.pipeline, variant=Variant(header=True)
        ),
    )
