"""Bootstrap, permutation test and the ADR-010 rule, on hand-computed and synthetic cases."""

import numpy as np
import pytest

from noterecall.eval.stats import (
    BootstrapCI,
    EvalStatsError,
    adopt,
    paired_bootstrap,
    paired_deltas,
    permutation_pvalue,
)


def test_paired_deltas_are_candidate_minus_baseline_in_id_order() -> None:
    deltas = paired_deltas({"b": 0.5, "a": 1.0}, {"a": 0.25, "b": 0.5})
    assert deltas == [0.75, 0.0]  # a: 1.0 - 0.25, b: 0.5 - 0.5


def test_paired_deltas_need_the_same_queries() -> None:
    with pytest.raises(EvalStatsError, match="different queries"):
        paired_deltas({"a": 1.0}, {"a": 1.0, "b": 0.0})


def test_constant_deltas_give_a_degenerate_interval() -> None:
    ci = paired_bootstrap([0.1] * 5, seed=1, resamples=500)
    assert (ci.n, ci.resamples, ci.seed) == (5, 500, 1)
    assert ci.mean == pytest.approx(0.1)
    assert ci.low == pytest.approx(0.1)
    assert ci.high == pytest.approx(0.1)


def test_two_point_interval_is_exactly_zero_to_one() -> None:
    # deltas [0, 1]: a resample of 2 has mean 0 (p 1/4), 0.5 (p 1/2) or 1 (p 1/4). Both tails
    # (2.5%) fall inside the 25% masses at 0 and 1, so the 95% interval is exactly [0, 1].
    ci = paired_bootstrap([0.0, 1.0], seed=7)
    assert (ci.low, ci.high, ci.mean, ci.resamples) == (0.0, 1.0, 0.5, 10_000)


def test_bootstrap_is_reproducible_and_seed_dependent() -> None:
    values = list(np.random.default_rng(0).normal(0.02, 0.1, size=60))
    a = paired_bootstrap(values, seed=5, resamples=2000)
    assert a == paired_bootstrap(values, seed=5, resamples=2000)
    assert a != paired_bootstrap(values, seed=6, resamples=2000)


def _trials(effect: float, trials: int) -> list[BootstrapCI]:
    """`trials` synthetic experiments: 60 paired deltas ~ Normal(effect, 0.1), seeded."""
    out: list[BootstrapCI] = []
    for seed in range(trials):
        sample = np.random.default_rng(1000 + seed).normal(effect, 0.1, size=60)
        out.append(paired_bootstrap(sample.tolist(), seed=seed, resamples=1000))
    return out


def test_known_effect_is_detected_and_covered() -> None:
    # true mean +0.05, sd 0.1, n 60: standard error 0.0129, so about 3.9 standard errors from 0
    # and the interval should exclude 0 almost always (power about 97%)
    cis = _trials(0.05, 300)
    power = sum(ci.low > 0 for ci in cis) / len(cis)
    coverage = sum(ci.low <= 0.05 <= ci.high for ci in cis) / len(cis)
    assert power >= 0.90
    assert 0.90 <= coverage <= 1.0  # nominal 95%


def test_no_effect_is_rarely_called_an_improvement() -> None:
    cis = _trials(0.0, 300)
    false_positive = sum(ci.low > 0 for ci in cis) / len(cis)
    assert false_positive <= 0.08  # one-sided nominal 2.5%


def test_empty_or_invalid_inputs_fail() -> None:
    with pytest.raises(EvalStatsError, match="no deltas"):
        paired_bootstrap([], seed=1)
    with pytest.raises(EvalStatsError, match="resamples"):
        paired_bootstrap([1.0], seed=1, resamples=0)
    with pytest.raises(EvalStatsError, match="no deltas"):
        permutation_pvalue([], seed=1)
    with pytest.raises(EvalStatsError, match="resamples"):
        permutation_pvalue([1.0], seed=1, resamples=0)


def test_permutation_pvalue_cases() -> None:
    # ten deltas of +1: only the all-plus and all-minus sign patterns reach |mean| = 1,
    # so the exact p is 2 / 1024 = 0.00195
    assert permutation_pvalue([1.0] * 10, seed=3) == pytest.approx(2 / 1024, abs=0.002)
    # all-zero deltas: every flip is as extreme as the observed mean, so p = 1
    assert permutation_pvalue([0.0] * 6, seed=3, resamples=100) == 1.0
    balanced = [1.0, -1.0] * 10
    assert permutation_pvalue(balanced, seed=3) > 0.5


def _ci(mean: float, low: float, high: float) -> BootstrapCI:
    return BootstrapCI(n=60, mean=mean, low=low, high=high, resamples=10_000, seed=1)


@pytest.mark.parametrize(
    ("ci", "free", "expected", "fragment"),
    [
        (_ci(0.05, 0.01, 0.09), False, True, "lower bound is above 0"),
        (_ci(0.05, 0.01, 0.09), True, True, "lower bound is above 0"),
        (_ci(0.02, -0.01, 0.05), False, False, "not free and the interval includes 0"),
        (_ci(0.02, -0.01, 0.05), True, True, "free, the interval includes 0"),
        (_ci(0.0, -0.02, 0.02), True, True, "free, the interval includes 0"),  # mean exactly 0
        (_ci(-0.01, -0.04, 0.02), True, False, "free, but the mean is negative"),
        (_ci(-0.05, -0.09, -0.01), True, False, "entirely below 0"),
        (_ci(-0.05, -0.09, -0.01), False, False, "entirely below 0"),
        (_ci(0.03, 0.0, 0.06), False, False, "not free"),  # lower bound exactly 0 is not > 0
    ],
)
def test_adopt_follows_adr_010(ci: BootstrapCI, free: bool, expected: bool, fragment: str) -> None:
    decision = adopt(ci, free)
    assert decision.adopt is expected
    assert fragment in decision.reason
