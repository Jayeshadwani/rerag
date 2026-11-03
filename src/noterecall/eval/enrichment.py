"""The M5 enrichment ablation and its pre-registered decisions. Implements ADR-010 with
ADR-003 (LLM clean) and ADR-007 (header, blurb).

Order and rules (decided before any enrichment ran, by the owner):

1. header: the M4 pipeline with `"{notebook} > {headings}"` in front of each indexed chunk, against
   the M4 pipeline. Reported, not decided: the header is deterministic and always on. If it hurts,
   that is a finding for a proposal, not something this code resolves;
2. blurb: header + LLM blurb against header alone, not free (it costs money): adopted only if the
   interval's lower bound is above 0;
3. clean: the best configuration so far, on LLM-cleaned notebooks, against the best so far, not
   free: adopted only if the lower bound is above 0, otherwise it ships off and the result is
   reported as negative.

All on dev Recall@10. The I/O is behind the `Evaluator` callable, as in `ablation.py`.
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
    StageOutcome,
    choose,
    compare,
)
from noterecall.retrieval.pipeline import PipelineSpec, Variant


class EnrichmentAblation(BaseModel):
    """Every stage, every configuration run, and the decided variant."""

    model_config = ConfigDict(frozen=True)

    stages: tuple[StageOutcome, ...]
    configs: tuple[ConfigResult, ...]
    chunk_cap: int
    pipeline: PipelineSpec
    variant: Variant
    reference: ConfigResult
    overall: Comparison | None = None


def header_note(comparison: Comparison) -> str:
    """What the header row means, stated plainly; a harmful header needs a proposal."""
    ci = comparison.ci
    if ci.high < 0:
        return (
            "the header HURTS (the interval is entirely below 0): this is evidence against the "
            "locked 'header always on' decision; see docs/proposals, nothing here changes it"
        )
    return "reported only: the header is deterministic and stays on (ADR-007)"


class _Runner:
    """Evaluates each variant once and applies the stage rules."""

    def __init__(
        self, evaluate: Evaluator, cap: int, spec: PipelineSpec, seed: int, chunker: Chunker
    ) -> None:
        self._evaluate = evaluate
        self._cap = cap
        self._spec = spec
        self._seed = seed
        self._chunker = chunker
        self.results: dict[RunKey, ConfigResult] = {}

    def run(self, variant: Variant) -> ConfigResult:
        key = RunKey(cap=self._cap, chunker=self._chunker, spec=self._spec, variant=variant)
        if key not in self.results:
            self.results[key] = self._evaluate(key)
        return self.results[key]

    def decide(self, stage: str, cand: ConfigResult, base: ConfigResult, free: bool) -> Comparison:
        return compare(Rule(stage=stage, metric=PRIMARY, free=free), cand, base, self._seed)

    def header(self, reference: ConfigResult) -> tuple[ConfigResult, StageOutcome]:
        with_header = self.run(Variant(header=True))
        comparison = self.decide("1 header (report only)", with_header, reference, True)
        comparison = comparison.model_copy(update={"report_only": True})
        note = header_note(comparison)
        outcome = StageOutcome(
            stage="1 header", comparisons=(comparison,), chosen=with_header.name, note=note
        )
        return with_header, outcome

    def blurb(self, base: ConfigResult) -> tuple[ConfigResult, StageOutcome]:
        blurbed = self.run(Variant(header=True, blurb=True))
        comparison = self.decide("2 blurb", blurbed, base, False)
        best = choose(base, [(blurbed, comparison)])
        return best, StageOutcome(stage="2 blurb", comparisons=(comparison,), chosen=best.name)

    def clean(self, base: ConfigResult) -> tuple[ConfigResult, StageOutcome]:
        cleaned = self.run(base.key.variant.model_copy(update={"clean": True}))
        comparison = self.decide("3 clean", cleaned, base, False)
        final = choose(base, [(cleaned, comparison)])
        return final, StageOutcome(stage="3 clean", comparisons=(comparison,), chosen=final.name)


def run_enrichment_ablation(
    evaluate: Evaluator, cap: int, spec: PipelineSpec, seed: int, chunker: Chunker = "structural"
) -> EnrichmentAblation:
    """Run header, blurb and clean in that order and apply the ADR-010 rules."""
    runner = _Runner(evaluate, cap, spec, seed, chunker)
    reference = runner.run(Variant())
    with_header, header_stage = runner.header(reference)
    best, blurb_stage = runner.blurb(with_header)
    final, clean_stage = runner.clean(best)
    overall = None
    if final is not reference:
        overall = runner.decide("overall (report only)", final, reference, True)
        overall = overall.model_copy(update={"report_only": True})
    return EnrichmentAblation(
        stages=(header_stage, blurb_stage, clean_stage),
        configs=tuple(runner.results.values()),
        chunk_cap=cap,
        pipeline=spec,
        variant=final.key.variant,
        reference=reference,
        overall=overall,
    )
