"""The header on/off row with hand-made scores (constant deltas give degenerate intervals)."""

from collections.abc import Mapping

from noterecall.eval.ablation import (
    ConfigResult,
    Evaluator,
    Latency,
    LatencyStat,
    RunKey,
    describe,
)
from noterecall.eval.header_ablation import header_note, run_header_ablation
from noterecall.retrieval.pipeline import PipelineSpec, Variant

QUERIES = [f"q{i}" for i in range(10)]
BASE = [0.2 + 0.05 * (i % 4) for i in range(10)]
SPEC = PipelineSpec(lexical="code", dense_model="base", rrf_k=10)
PREFIX = "structural-512 | bm25-code + dense:base + rrf10"
HEADER = f"{PREFIX} +header"
STAT = LatencyStat(p50=1.0, p95=2.0)
LATENCY = Latency(lexical=STAT, dense=STAT, fusion=STAT, rerank=STAT, total=STAT)


def fake(offsets: Mapping[str, float | list[float]]) -> Evaluator:
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


def test_describe_marks_the_header_and_leaves_the_m4_name_alone() -> None:
    plain = RunKey(cap=512, chunker="structural", spec=SPEC)
    assert describe(plain) == PREFIX
    assert describe(plain.model_copy(update={"variant": Variant(header=True)})) == HEADER


def test_a_run_key_without_a_variant_still_loads_as_the_m4_pipeline() -> None:
    key = RunKey.model_validate_json(
        '{"cap": 512, "chunker": "structural", "spec": {"lexical": "code"}}'
    )
    assert key.variant == Variant()


def test_the_row_compares_with_against_without_and_is_report_only() -> None:
    result = run_header_ablation(fake({HEADER: 0.04}), 512, SPEC, seed=7)
    row = result.comparison
    assert (row.candidate, row.baseline) == (HEADER, PREFIX)
    assert abs(row.ci.mean - 0.04) < 1e-12  # every delta is 0.04, so the interval is that number
    assert row.report_only
    assert row.metric == "recall@10"
    assert result.with_header.key.variant == Variant(header=True)
    assert result.without_header.key.variant == Variant()
    assert "stays on" in result.note


def test_a_header_that_hurts_is_reported_as_evidence_against_the_locked_decision() -> None:
    result = run_header_ablation(fake({HEADER: -0.03}), 512, SPEC, seed=7)
    assert result.comparison.decision.adopt is False
    assert "HURTS" in result.note
    assert "proposal" in result.note


def test_a_noisy_negative_effect_is_not_called_harmful() -> None:
    deltas = [-0.02 if i % 2 == 0 else 0.015 for i in range(10)]  # mean -0.0025, spans 0
    result = run_header_ablation(fake({HEADER: deltas}), 512, SPEC, seed=7)
    assert result.comparison.ci.mean < 0 < result.comparison.ci.high
    assert "HURTS" not in header_note(result.comparison)
    assert "HURTS" not in result.note
