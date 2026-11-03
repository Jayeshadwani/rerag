"""The M5 stages with hand-made scores: header (reported), blurb (not free), clean (not free)."""

from collections.abc import Mapping

from noterecall.eval.ablation import (
    ConfigResult,
    Evaluator,
    Latency,
    LatencyStat,
    RunKey,
    describe,
)
from noterecall.eval.enrichment import EnrichmentAblation, header_note, run_enrichment_ablation
from noterecall.retrieval.pipeline import PipelineSpec, Variant

QUERIES = [f"q{i}" for i in range(10)]
BASE = [0.2 + 0.05 * (i % 4) for i in range(10)]
SPEC = PipelineSpec(lexical="code", dense_model="base", rrf_k=10)
PREFIX = "structural-512 | bm25-code + dense:base + rrf10"
HEADER = f"{PREFIX} +header"
BLURB = f"{PREFIX} +header +blurb"
CLEAN = f"{PREFIX} +header +clean"
BLURB_CLEAN = f"{PREFIX} +header +blurb +clean"
STAT = LatencyStat(p50=1.0, p95=2.0)
LATENCY = Latency(lexical=STAT, dense=STAT, fusion=STAT, rerank=STAT, total=STAT)


def fake(offsets: Mapping[str, float | list[float]]) -> Evaluator:
    """A configuration scores BASE plus the offset listed under its name (default 0)."""

    def evaluate(key: RunKey) -> ConfigResult:
        name = describe(key)
        offset = offsets.get(name, 0.0)
        per_query = {
            q: {"recall@10": BASE[i] + (offset[i] if isinstance(offset, list) else offset)}
            for i, q in enumerate(QUERIES)
        }
        means = {"recall@10": sum(s["recall@10"] for s in per_query.values()) / len(per_query)}
        return ConfigResult(name=name, key=key, means=means, per_query=per_query, latency=LATENCY)

    return evaluate


def run(offsets: Mapping[str, float | list[float]]) -> EnrichmentAblation:
    return run_enrichment_ablation(fake(offsets), 512, SPEC, seed=7)


def test_describe_marks_each_enrichment_and_leaves_the_m4_name_alone() -> None:
    plain = RunKey(cap=512, chunker="structural", spec=SPEC)
    assert describe(plain) == PREFIX
    assert describe(plain.model_copy(update={"variant": Variant(header=True, clean=True)})) == (
        f"{PREFIX} +header +clean"
    )


def test_a_run_key_without_a_variant_still_loads_as_the_m4_pipeline() -> None:
    key = RunKey.model_validate_json(
        '{"cap": 512, "chunker": "structural", "spec": {"lexical": "code"}}'
    )
    assert key.variant == Variant()


def test_everything_that_helps_is_adopted_and_the_stages_run_in_order() -> None:
    ablation = run({HEADER: 0.01, BLURB: 0.06, BLURB_CLEAN: 0.10})
    assert ablation.variant == Variant(header=True, blurb=True, clean=True)
    assert [s.stage for s in ablation.stages] == ["1 header", "2 blurb", "3 clean"]
    header, blurb, clean = (s.comparisons[0] for s in ablation.stages)
    assert (header.candidate, header.baseline) == (HEADER, PREFIX)
    assert (blurb.candidate, blurb.baseline) == (BLURB, HEADER)  # the header is the base
    assert (clean.candidate, clean.baseline) == (BLURB_CLEAN, BLURB)  # best so far
    # every delta is constant, so the interval is exactly that number: 0.06 - 0.01, 0.10 - 0.06
    assert abs(blurb.ci.mean - 0.05) < 1e-12
    assert abs(clean.ci.mean - 0.04) < 1e-12
    assert blurb.decision.adopt
    assert clean.decision.adopt
    assert (header.free, blurb.free, clean.free) == (True, False, False)
    assert header.report_only
    assert not blurb.report_only
    assert not clean.report_only
    assert ablation.overall is not None
    assert ablation.overall.report_only
    assert ablation.overall.baseline == PREFIX
    assert ablation.overall.candidate == BLURB_CLEAN


def test_a_blurb_whose_interval_includes_zero_is_not_adopted_because_it_is_not_free() -> None:
    deltas = [0.02 if i % 2 == 0 else -0.01 for i in range(10)]  # mean +0.005, noisy
    ablation = run({BLURB: deltas})
    blurb = ablation.stages[1].comparisons[0]
    assert blurb.ci.low < 0 < blurb.ci.high
    assert blurb.ci.mean > 0
    assert not blurb.decision.adopt  # a free component would be adopted here
    assert not ablation.variant.blurb
    assert ablation.stages[1].chosen == HEADER


def test_clean_is_judged_against_the_best_so_far_and_ships_off_when_it_does_not_help() -> None:
    ablation = run({BLURB: 0.05, BLURB_CLEAN: 0.04})
    clean = ablation.stages[2].comparisons[0]
    assert clean.baseline == BLURB
    assert clean.candidate == BLURB_CLEAN
    assert not clean.decision.adopt
    assert ablation.variant == Variant(header=True, blurb=True)
    assert ablation.stages[2].chosen == BLURB


def test_clean_is_tried_on_the_header_alone_when_the_blurb_was_not_adopted() -> None:
    ablation = run({CLEAN: 0.05})
    clean = ablation.stages[2].comparisons[0]
    assert (clean.candidate, clean.baseline) == (CLEAN, HEADER)
    assert ablation.variant == Variant(header=True, clean=True)


def test_the_header_stays_on_even_when_it_hurts_and_the_row_says_so() -> None:
    ablation = run({HEADER: -0.03})
    header = ablation.stages[0]
    assert ablation.variant.header  # carried forward: it is a locked decision, not decided here
    assert header.comparisons[0].decision.adopt is False
    assert "HURTS" in header.note
    assert "proposal" in header.note
    assert "HURTS" not in header_note(run({HEADER: 0.02}).stages[0].comparisons[0])


def test_every_variant_is_evaluated_once_and_reported() -> None:
    calls: list[str] = []
    inner = fake({})

    def counting(key: RunKey) -> ConfigResult:
        calls.append(describe(key))
        return inner(key)

    ablation = run_enrichment_ablation(counting, 512, SPEC, seed=7)
    assert len(calls) == len(set(calls))  # no configuration is run twice
    assert [c.name for c in ablation.configs] == calls
    assert ablation.reference.name == PREFIX
