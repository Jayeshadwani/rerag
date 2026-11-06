"""Stage-1 abstention and its calibration. Implements ADR-011 as amended by proposal 006.

A query is refused, with no LLM call, only if its top dense cosine similarity is below `tau_cos`
**and** its normalized top BM25 score is below `tau_bm25` (the normalization is made by the
pipeline: top score divided by the number of query tokens). `calibrate` applies the pre-registered
rule of proposal 006 to the dev queries: candidate thresholds are the midpoints between
consecutive distinct values plus one value below the smallest and one above the largest; a pair is
feasible if its false-refusal rate on answerable queries is at most the bound (checked in
integers); the feasible pair that refuses the most unanswerable queries wins; ties go to the lowest
`tau_cos`, then the lowest `tau_bm25`. Everything is pure; the full grid is returned.
"""

from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from noterecall.retrieval.pipeline import AbstentionSignals


class Thresholds(BaseModel):
    """The two stage-1 thresholds."""

    model_config = ConfigDict(frozen=True)

    cosine: float
    bm25: float


class CalibrationPoint(NamedTuple):
    """One dev query: whether it is unanswerable, and its stage-1 signals."""

    query_id: str
    unanswerable: bool
    signals: AbstentionSignals


class RefusalMetrics(BaseModel):
    """Counts first; rates and precision/F1 are derived (precision and F1 are for reference)."""

    model_config = ConfigDict(frozen=True)

    answerable: int
    unanswerable: int
    false_refusals: int
    caught: int
    false_refusal_rate: float | None
    precision: float | None
    recall: float | None
    f1: float | None


class GridRow(BaseModel):
    """One threshold pair and what it does to the dev queries."""

    model_config = ConfigDict(frozen=True)

    tau_cos: float
    tau_bm25: float
    caught: int
    false_refusals: int


class Choice(BaseModel):
    """A chosen pair with its dev metrics."""

    model_config = ConfigDict(frozen=True)

    thresholds: Thresholds
    metrics: RefusalMetrics


class Calibration(BaseModel):
    """The dev calibration: the grid and the two stage-1 rows of proposal 006."""

    model_config = ConfigDict(frozen=True)

    max_false_refusal_pct: int
    grid: tuple[GridRow, ...]
    cosine_only: Choice
    cosine_bm25: Choice


class CalibrationError(Exception):
    """Raised when the dev points cannot be calibrated on (no answerable or no unanswerable)."""


def refuse(signals: AbstentionSignals, thresholds: Thresholds) -> bool:
    """Stage 1: refuse only if both signals are below their thresholds."""
    return signals.top_cosine < thresholds.cosine and signals.bm25_norm < thresholds.bm25


def candidates(values: Iterable[float]) -> list[float]:
    """Midpoints between consecutive distinct values, plus one below the smallest and one above
    the largest (so every axis can be made never-true and always-true)."""
    distinct = sorted(set(values))
    mids = [(a + b) / 2 for a, b in pairwise(distinct)]
    return [distinct[0] - 1.0, *mids, distinct[-1] + 1.0]


def refusal_metrics(decisions: Iterable[tuple[bool, bool]]) -> RefusalMetrics:
    """Metrics from (unanswerable, refused) pairs."""
    answerable = unanswerable = false_refusals = caught = 0
    for is_unanswerable, refused in decisions:
        if is_unanswerable:
            unanswerable += 1
            caught += refused
        else:
            answerable += 1
            false_refusals += refused
    refusals = caught + false_refusals
    precision = caught / refusals if refusals else None
    recall = caught / unanswerable if unanswerable else None
    f1 = 2 * caught / (refusals + unanswerable) if refusals + unanswerable else None
    return RefusalMetrics(
        answerable=answerable,
        unanswerable=unanswerable,
        false_refusals=false_refusals,
        caught=caught,
        false_refusal_rate=false_refusals / answerable if answerable else None,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def metrics_at(points: Sequence[CalibrationPoint], thresholds: Thresholds) -> RefusalMetrics:
    """Stage-1 metrics of a threshold pair on a set of queries."""
    return refusal_metrics((p.unanswerable, refuse(p.signals, thresholds)) for p in points)


def _best(rows: Iterable[GridRow], answerable: int, max_pct: int) -> GridRow:
    feasible = [r for r in rows if r.false_refusals * 100 <= max_pct * answerable]
    return min(feasible, key=lambda r: (-r.caught, r.tau_cos, r.tau_bm25))


def calibrate(points: Sequence[CalibrationPoint], max_false_refusal_pct: int) -> Calibration:
    """Apply the pre-registered rule of proposal 006 to the dev points."""
    answerable = sum(1 for p in points if not p.unanswerable)
    if answerable == 0 or answerable == len(points):
        raise CalibrationError("calibration needs both answerable and unanswerable queries")
    cos_values = candidates(p.signals.top_cosine for p in points)
    bm_values = candidates(p.signals.bm25_norm for p in points)
    grid = []
    for tau_cos in cos_values:
        for tau_bm in bm_values:
            m = metrics_at(points, Thresholds(cosine=tau_cos, bm25=tau_bm))
            grid.append(GridRow(tau_cos=tau_cos, tau_bm25=tau_bm, caught=m.caught,
                                false_refusals=m.false_refusals))  # fmt: skip
    always_true = bm_values[-1]
    only_cos = _best(
        (r for r in grid if r.tau_bm25 == always_true), answerable, max_false_refusal_pct
    )
    both = _best(grid, answerable, max_false_refusal_pct)

    def choice(row: GridRow) -> Choice:
        t = Thresholds(cosine=row.tau_cos, bm25=row.tau_bm25)
        return Choice(thresholds=t, metrics=metrics_at(points, t))

    return Calibration(
        max_false_refusal_pct=max_false_refusal_pct,
        grid=tuple(grid),
        cosine_only=choice(only_cos),
        cosine_bm25=choice(both),
    )
