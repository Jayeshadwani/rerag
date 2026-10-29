"""Paired bootstrap, sign-flip permutation test and the ADR-010 decision rule.

ADR-010: for component X vs a baseline, take per-query metric deltas on dev, run a paired
bootstrap (10,000 resamples) and a 95% confidence interval. Adopt X if the interval's lower bound
is above 0, or if X is free (no latency or cost) and the interval includes 0 with a non-negative
mean. Everything is seeded, so a result file can be reproduced exactly.
"""

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from noterecall.ingest.stats import GitState

BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE = 0.95


class EvalStatsError(Exception):
    """Raised when a statistic is asked for on unusable input."""


class BootstrapCI(BaseModel):
    """Mean of the paired deltas and its percentile bootstrap interval."""

    model_config = ConfigDict(frozen=True)

    n: int
    mean: float
    low: float
    high: float
    resamples: int
    seed: int


class Decision(BaseModel):
    """The ADR-010 outcome and the clause that produced it."""

    model_config = ConfigDict(frozen=True)

    adopt: bool
    reason: str


def paired_deltas(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> list[float]:
    """candidate - baseline per query id, in sorted id order. Both must cover the same queries."""
    if set(candidate) != set(baseline):
        raise EvalStatsError("candidate and baseline scores cover different queries")
    return [candidate[i] - baseline[i] for i in sorted(candidate)]


def _array(deltas: Sequence[float]) -> np.ndarray:
    if len(deltas) == 0:
        raise EvalStatsError("no deltas to resample")
    return np.asarray(deltas, dtype=np.float64)


def paired_bootstrap(
    deltas: Sequence[float], seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> BootstrapCI:
    """Percentile bootstrap CI of the mean delta: resample queries with replacement."""
    values = _array(deltas)
    if resamples < 1:
        raise EvalStatsError("resamples must be at least 1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1.0 - CONFIDENCE) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return BootstrapCI(
        n=len(values),
        mean=float(values.mean()),
        low=float(low),
        high=float(high),
        resamples=resamples,
        seed=seed,
    )


def permutation_pvalue(
    deltas: Sequence[float], seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> float:
    """Two-sided paired sign-flip test of mean delta = 0, with the +1 correction."""
    values = _array(deltas)
    if resamples < 1:
        raise EvalStatsError("resamples must be at least 1")
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(resamples, len(values))) * 2 - 1
    flipped = (signs * values).mean(axis=1)
    extreme = int(np.sum(np.abs(flipped) >= abs(values.mean()) - 1e-12))
    return (extreme + 1) / (resamples + 1)


def adopt(ci: BootstrapCI, free: bool) -> Decision:
    """The ADR-010 decision rule, applied mechanically."""
    if ci.low > 0:
        return Decision(adopt=True, reason="the interval's lower bound is above 0")
    includes_zero = ci.low <= 0 <= ci.high
    if free and includes_zero and ci.mean >= 0:
        return Decision(
            adopt=True, reason="free, the interval includes 0 and the mean is not negative"
        )
    if free and includes_zero:
        return Decision(adopt=False, reason="free, but the mean is negative")
    if includes_zero:
        return Decision(adopt=False, reason="not free and the interval includes 0")
    return Decision(adopt=False, reason="the interval is entirely below 0")


class SimulationSpec(BaseModel):
    """One synthetic setting: n paired deltas ~ Normal(effect, sd), repeated `trials` times."""

    model_config = ConfigDict(frozen=True)

    effect: float
    sd: float
    n: int
    trials: int
    resamples: int


class SimulationResult(BaseModel):
    """How the bootstrap behaves on synthetic data with a known true effect."""

    model_config = ConfigDict(frozen=True)

    spec: SimulationSpec
    excludes_zero_above: float
    covers_true_effect: float


def simulate_bootstrap(spec: SimulationSpec, seed: int) -> SimulationResult:
    """Run the synthetic experiments in `spec`.

    Reports the share of intervals whose lower bound is above 0 (power when effect > 0, the
    one-sided false-positive rate when effect = 0) and the share that contain the true effect.
    """
    if spec.trials < 1:
        raise EvalStatsError("trials must be at least 1")
    above = covered = 0
    for trial in range(spec.trials):
        sample = np.random.default_rng([seed, trial]).normal(spec.effect, spec.sd, size=spec.n)
        ci = paired_bootstrap(sample.tolist(), seed=trial, resamples=spec.resamples)
        above += ci.low > 0
        covered += ci.low <= spec.effect <= ci.high
    return SimulationResult(
        spec=spec,
        excludes_zero_above=above / spec.trials,
        covers_true_effect=covered / spec.trials,
    )


class BootstrapValidation(BaseModel):
    """The contents of `results/bootstrap_validation.json`: synthetic data only."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1]
    git: GitState
    seed: int
    simulations: tuple[SimulationResult, ...]


def validate_bootstrap(
    specs: Sequence[SimulationSpec], seed: int, git: GitState
) -> BootstrapValidation:
    """Simulate every setting; no private data is involved."""
    return BootstrapValidation(
        schema_version=1,
        git=git,
        seed=seed,
        simulations=tuple(simulate_bootstrap(spec, seed) for spec in specs),
    )
