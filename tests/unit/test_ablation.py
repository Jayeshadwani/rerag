"""The staged decisions with hand-made scores. Constant per-query deltas make every interval
degenerate (low = high = the delta), so each ADR-010 outcome is known in advance."""

import pytest

from noterecall.eval.ablation import (
    AblationError,
    AblationParams,
    Comparison,
    ConfigResult,
    Evaluator,
    Latency,
    LatencyStat,
    Rule,
    RunKey,
    choose,
    compare,
    describe,
    percentile_stat,
    run_ablation,
    summarize_latency,
)
from noterecall.retrieval.pipeline import PipelineSpec, StageTimes

QUERIES = [f"q{i}" for i in range(10)]
BASE = [0.2 + 0.05 * (i % 4) for i in range(10)]  # varied, so deltas are not trivially aligned
METRICS = ("recall@10", "recall@tokens=100")
PARAMS = AblationParams(
    caps=(256, 512, 1024),
    baseline_cap=512,
    budget_metric="recall@tokens=100",
    embedding_models=("small", "base", "other"),
    rrf_default=60,
    rrf_challengers=(10, 30),
    latency_budget_ms=1000.0,
    seed=7,
)


def latency(total_p95: float = 100.0) -> Latency:
    stat = LatencyStat(p50=total_p95 / 2, p95=total_p95)
    return Latency(lexical=stat, dense=stat, fusion=stat, rerank=stat, total=stat)


def fake(offsets: dict[str, float], p95: float = 100.0) -> Evaluator:
    """An evaluator: a configuration's score is BASE plus the offset listed under its name."""

    def evaluate(key: RunKey) -> ConfigResult:
        name = describe(key)
        offset = offsets.get(name, 0.0)
        per_query = {q: dict.fromkeys(METRICS, BASE[i] + offset) for i, q in enumerate(QUERIES)}
        means = {m: sum(s[m] for s in per_query.values()) / len(per_query) for m in METRICS}
        return ConfigResult(
            name=name, key=key, means=means, per_query=per_query, latency=latency(p95)
        )

    return evaluate


def result(name: str, offset: float) -> ConfigResult:
    """A configuration whose every query scores BASE plus the offset."""
    key = RunKey(cap=512, chunker="structural", spec=PipelineSpec(lexical="code"))
    per_query = {q: dict.fromkeys(METRICS, BASE[i] + offset) for i, q in enumerate(QUERIES)}
    means = {m: sum(s[m] for s in per_query.values()) / len(per_query) for m in METRICS}
    return ConfigResult(name=name, key=key, means=means, per_query=per_query, latency=latency())


GOOD = {
    "structural-256 | bm25-code": -0.10,  # a smaller cap is worse
    "structural-1024 | bm25-code": -0.02,  # a larger one is worse too
    "fixed-256 | bm25-code": -0.05,
    "fixed-512 | bm25-code": -0.04,
    "fixed-1024 | bm25-code": -0.03,
    "structural-512 | bm25-code": 0.10,  # the code tokenizer beats plain (0)
    "structural-512 | dense:base": 0.02,  # base beats small (0) clearly, 'other' does not
    "structural-512 | dense:other": -0.01,
    "structural-512 | bm25-code + dense:base + rrf60": 0.20,
    "structural-512 | bm25-code + dense:base + rrf10": 0.19,
    "structural-512 | bm25-code + dense:base + rrf30": 0.23,
    "structural-512 | bm25-code + dense:base + rrf30 + rerank": 0.30,
}


def test_every_stage_adopts_when_the_data_say_so() -> None:
    ablation = run_ablation(fake(GOOD), PARAMS)
    by_stage = {s.stage: s for s in ablation.stages}
    # the chunking stage compares structural with fixed at each cap, then the other caps with 512
    chunking = by_stage["0 chunking"]
    assert [c.candidate for c in chunking.comparisons[:3]] == [
        "structural-256 | bm25-code",
        "structural-512 | bm25-code",
        "structural-1024 | bm25-code",
    ]
    assert all(c.decision.adopt for c in chunking.comparisons[:3] if "512" in c.candidate)
    assert chunking.chosen == "structural-512 | bm25-code"  # neither challenger beats 512
    assert ablation.chunk_cap == 512
    assert by_stage["1 tokenizer"].chosen == "structural-512 | bm25-code"
    assert by_stage["2 embedding"].chosen == "structural-512 | dense:base"
    assert by_stage["3 hybrid"].chosen == "structural-512 | bm25-code + dense:base + rrf60"
    assert by_stage["4 rrf k"].chosen == "structural-512 | bm25-code + dense:base + rrf30"
    assert by_stage["5 rerank"].chosen.endswith("rrf30 + rerank")
    assert ablation.pipeline == PipelineSpec(
        lexical="code", dense_model="base", rrf_k=30, rerank=True
    )
    assert ablation.reference.name == "structural-512 | bm25-code"


def test_rule_details_for_a_not_free_component() -> None:
    ablation = run_ablation(fake(GOOD), PARAMS)
    embedding = next(s for s in ablation.stages if s.stage == "2 embedding")
    base, other = embedding.comparisons
    assert base.candidate == "structural-512 | dense:base"
    assert base.baseline == "structural-512 | dense:small"
    assert (base.ci.low, base.ci.high) == pytest.approx((0.02, 0.02))
    assert base.decision.adopt  # lower bound above 0
    assert not other.decision.adopt  # the mean is negative and the component is not free
    assert not base.free
    assert base.metric == "recall@10"


def test_every_configuration_run_is_reported() -> None:
    ablation = run_ablation(fake(GOOD), PARAMS)
    names = {c.name for c in ablation.configs}
    assert "structural-512 | bm25-code + dense:base + rrf10" in names  # the rejected k too
    assert "fixed-512 | bm25-code" in names
    assert len(names) == len(ablation.configs)  # each configuration ran once


def test_losers_everywhere_keep_the_baselines_and_skip_rrf_tuning() -> None:
    offsets = {
        "structural-512 | bm25-code": -0.05,  # worse than plain: stay with plain
        "structural-256 | bm25-code": -0.15,  # (the other caps stay worse than 512)
        "structural-1024 | bm25-code": -0.15,
        "structural-512 | dense:base": -0.02,  # not better than small
        "structural-512 | dense:other": -0.02,
        "structural-512 | bm25-plain + dense:small + rrf60": -0.01,  # hybrid worse than plain
    }
    ablation = run_ablation(fake(offsets), PARAMS)
    by_stage = {s.stage: s for s in ablation.stages}
    assert by_stage["1 tokenizer"].chosen == "structural-512 | bm25-plain"
    assert by_stage["2 embedding"].chosen == "structural-512 | dense:small"
    assert by_stage["3 hybrid"].chosen == "structural-512 | bm25-plain"
    assert by_stage["4 rrf k"].comparisons == ()
    assert "skipped" in by_stage["4 rrf k"].note
    assert by_stage["5 rerank"].chosen == "structural-512 | bm25-plain"  # rerank not better
    assert ablation.pipeline == PipelineSpec(lexical="plain")


def test_a_better_dense_model_than_bm25_is_the_single_retriever_to_beat() -> None:
    offsets = {
        "structural-256 | bm25-code": -0.10,  # keep 512 as the chunk cap
        "structural-1024 | bm25-code": -0.10,
        "structural-512 | dense:small": 0.10,  # dense alone beats bm25 (the free tie keeps code)
        "structural-512 | bm25-code + dense:small + rrf60": 0.18,
    }
    ablation = run_ablation(fake(offsets), PARAMS)
    hybrid = next(s for s in ablation.stages if s.stage == "3 hybrid")
    assert hybrid.comparisons[0].baseline == "structural-512 | dense:small"
    assert hybrid.chosen == "structural-512 | bm25-code + dense:small + rrf60"


def test_rerank_is_rejected_when_the_latency_budget_is_blown() -> None:
    offsets = dict(GOOD)
    slow = fake(offsets, p95=1500.0)
    ablation = run_ablation(slow, PARAMS)
    stage = next(s for s in ablation.stages if s.stage == "5 rerank")
    (comparison,) = stage.comparisons
    assert not comparison.decision.adopt
    assert comparison.ci.low > 0  # it was better ...
    assert "over budget" in comparison.decision.reason  # ... but too slow
    assert comparison.gate == "p95 total latency 1500 ms vs budget 1000 ms"
    assert not ablation.pipeline.rerank


def test_rerank_gate_is_recorded_even_when_the_budget_holds() -> None:
    ablation = run_ablation(fake(GOOD, p95=400.0), PARAMS)
    stage = next(s for s in ablation.stages if s.stage == "5 rerank")
    assert stage.comparisons[0].gate == "p95 total latency 400 ms vs budget 1000 ms"
    assert ablation.pipeline.rerank


def test_a_free_tie_is_adopted_by_the_rule() -> None:
    # ADR-010 literally: free, interval [0, 0] includes 0, mean 0 is not negative -> adopt
    offsets = {"structural-1024 | bm25-code": 0.0, "structural-256 | bm25-code": -0.1}
    ablation = run_ablation(fake(offsets), PARAMS)
    chunking = next(s for s in ablation.stages if s.stage == "0 chunking")
    assert ablation.chunk_cap == 1024
    assert chunking.chosen == "structural-1024 | bm25-code"


def test_compare_is_candidate_minus_baseline() -> None:
    rule = Rule(stage="s", metric="recall@10", free=True)
    up = result("up", 0.1)
    down = result("down", 0.0)
    forward = compare(rule, up, down, seed=1)
    backward = compare(rule, down, up, seed=1)
    assert forward.ci.mean == pytest.approx(0.1)
    assert backward.ci.mean == pytest.approx(-0.1)
    assert forward.decision.adopt
    assert not backward.decision.adopt


def comparison_with(mean: float, adopted: bool) -> Comparison:
    rule = Rule(stage="s", metric="recall@10", free=False)
    base, cand = result("b", 0.0), result("c", mean)
    return compare(rule, cand, base, seed=1).model_copy(
        update={
            "decision": compare(rule, cand, base, seed=1).decision.model_copy(
                update={"adopt": adopted}
            )
        }
    )


def test_choose_takes_the_best_adopted_and_prefers_the_earlier_on_ties() -> None:
    base = result("base", 0.0)
    first, second, third = result("first", 0.1), result("second", 0.2), result("third", 0.2)
    pairs = [
        (first, comparison_with(0.1, True)),
        (second, comparison_with(0.2, True)),
        (third, comparison_with(0.2, True)),
    ]
    assert choose(base, pairs) is second  # best mean; the earlier of the tied pair
    pairs[1] = (second, comparison_with(0.2, False))
    assert choose(base, pairs) is third
    assert choose(base, [(first, comparison_with(0.1, False))]) is base
    assert choose(base, []) is base


def test_percentiles_are_nearest_rank() -> None:
    stat = percentile_stat([float(v) for v in range(1, 21)])  # 1..20
    assert stat.p50 == 10.5  # median of an even count
    assert stat.p95 == 19.0  # ceil(0.95 * 20) = 19th value
    assert percentile_stat([5.0]) == LatencyStat(p50=5.0, p95=5.0)
    assert percentile_stat([3.0, 1.0, 2.0]).p50 == 2.0  # odd count, unsorted input


def test_latency_summary_covers_every_stage() -> None:
    times = [
        StageTimes(lexical_ms=1, dense_ms=2, fusion_ms=3, rerank_ms=4, total_ms=10 * (i + 1))
        for i in range(4)
    ]
    summary = summarize_latency(times)
    assert summary.total.p50 == 25.0  # 10, 20, 30, 40
    assert summary.total.p95 == 40.0
    assert (summary.lexical.p50, summary.dense.p50, summary.fusion.p50, summary.rerank.p50) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )


def test_describe_names_every_part() -> None:
    key = RunKey(
        cap=256,
        chunker="fixed",
        spec=PipelineSpec(lexical="plain", dense_model="m", rrf_k=10, rerank=True),
    )
    assert describe(key) == "fixed-256 | bm25-plain + dense:m + rrf10 + rerank"


def test_an_embedding_stage_result_without_a_dense_model_is_a_loud_error() -> None:
    honest = fake({})

    def broken(key: RunKey) -> ConfigResult:
        done = honest(key)
        if key.spec.dense_model is not None and key.spec.lexical is None:
            wrong = RunKey(cap=key.cap, chunker=key.chunker, spec=PipelineSpec(lexical="code"))
            return done.model_copy(update={"key": wrong})
        return done

    with pytest.raises(AblationError, match="without a dense model"):
        run_ablation(broken, PARAMS)
