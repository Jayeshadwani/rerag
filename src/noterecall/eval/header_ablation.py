"""The M5 header ablation. Implements ADR-007 (deterministic header) under ADR-010.

One comparison, on dev Recall@10: the M4 pipeline with the context header in front of each indexed
chunk against the same pipeline without it. It is **reported, not decided on**: ADR-007 locks the
header on. If the header turned out to hurt, that would be evidence against a locked decision and
the answer is a proposal, not a change here. The LLM blurb and LLM clean are not run (proposal 005),
so they have no rows. The I/O is behind the `Evaluator` callable, as in `ablation.py`.
"""

from pydantic import BaseModel, ConfigDict

from noterecall.eval.ablation import (
    PRIMARY,
    Chunker,
    Comparison,
    ConfigResult,
    Evaluator,
    Rule,
    RunKey,
    compare,
)
from noterecall.retrieval.pipeline import PipelineSpec, Variant


class HeaderAblation(BaseModel):
    """Both configurations, the comparison, and what it says in words."""

    model_config = ConfigDict(frozen=True)

    chunk_cap: int
    pipeline: PipelineSpec
    without_header: ConfigResult
    with_header: ConfigResult
    comparison: Comparison
    note: str


def header_note(comparison: Comparison) -> str:
    """What the row means, stated plainly; a harmful header needs a proposal."""
    if comparison.ci.high < 0:
        return (
            "the header HURTS (the interval is entirely below 0): this is evidence against the "
            "locked 'header always on' decision; see docs/proposals, nothing here changes it"
        )
    return "reported only: the header is deterministic and stays on (ADR-007)"


def run_header_ablation(
    evaluate: Evaluator, cap: int, spec: PipelineSpec, seed: int, chunker: Chunker = "structural"
) -> HeaderAblation:
    """Evaluate the pipeline with and without the header and compare them (report only)."""

    def run(variant: Variant) -> ConfigResult:
        return evaluate(RunKey(cap=cap, chunker=chunker, spec=spec, variant=variant))

    without, with_header = run(Variant()), run(Variant(header=True))
    rule = Rule(stage="header on vs off (report only)", metric=PRIMARY, free=True)
    comparison = compare(rule, with_header, without, seed).model_copy(update={"report_only": True})
    return HeaderAblation(
        chunk_cap=cap,
        pipeline=spec,
        without_header=without,
        with_header=with_header,
        comparison=comparison,
        note=header_note(comparison),
    )
