"""Stage-1 abstention and the pre-registered calibration, with hand-computed expectations."""

import pytest

from noterecall.generate.abstain import (
    CalibrationError,
    CalibrationPoint,
    Thresholds,
    calibrate,
    candidates,
    metrics_at,
    refusal_metrics,
    refuse,
)
from noterecall.retrieval.pipeline import AbstentionSignals


def sig(cos: float, bm: float) -> AbstentionSignals:
    return AbstentionSignals(top_cosine=cos, bm25_norm=bm)


def test_refuse_needs_both_signals_below_their_thresholds() -> None:
    t = Thresholds(cosine=0.7, bm25=2.0)
    assert refuse(sig(0.5, 1.0), t)
    assert not refuse(sig(0.8, 1.0), t)  # the cosine is high enough to answer
    assert not refuse(sig(0.5, 3.0), t)  # the keyword match is strong enough to answer
    assert not refuse(sig(0.7, 1.0), t)  # equal is not below
    assert not refuse(sig(0.5, 2.0), t)


def test_candidates_are_midpoints_plus_one_below_and_one_above() -> None:
    # distinct values 1, 2, 4 -> midpoints 1.5 and 3; below 1 - 1 = 0; above 4 + 1 = 5
    assert candidates([4, 1, 2, 2, 1]) == [0.0, 1.5, 3.0, 5.0]
    assert candidates([7.0]) == [6.0, 8.0]


def test_refusal_metrics_hand_computed() -> None:
    # 3 answerable (1 wrongly refused), 2 unanswerable (1 caught); 2 refusals in all
    m = refusal_metrics(
        [(True, True), (True, False), (False, True), (False, False), (False, False)]
    )
    assert (m.answerable, m.unanswerable, m.false_refusals, m.caught) == (3, 2, 1, 1)
    assert m.false_refusal_rate == pytest.approx(1 / 3)
    assert m.precision == 0.5  # 1 caught of 2 refusals
    assert m.recall == 0.5  # 1 caught of 2 unanswerable
    assert m.f1 == 0.5  # 2 * 1 / (2 refusals + 2 unanswerable)


def test_refusal_metrics_with_no_refusals_has_no_precision() -> None:
    m = refusal_metrics([(True, False), (False, False)])
    assert (m.precision, m.recall, m.f1, m.false_refusal_rate) == (None, 0.0, 0.0, 0.0)
    only_answerable = refusal_metrics([(False, False)])
    assert (only_answerable.recall, only_answerable.f1) == (None, None)


# Answerable: a1 (0.9, 5) a2 (0.8, 4) a3 (0.7, 3) a4 (0.6, 1)
# Unanswerable: u1 (0.55, 0.5) u2 (0.65, 0.2) u3 (0.85, 0.1)
POINTS = [
    CalibrationPoint("a1", False, sig(0.9, 5)),
    CalibrationPoint("a2", False, sig(0.8, 4)),
    CalibrationPoint("a3", False, sig(0.7, 3)),
    CalibrationPoint("a4", False, sig(0.6, 1)),
    CalibrationPoint("u1", True, sig(0.55, 0.5)),
    CalibrationPoint("u2", True, sig(0.65, 0.2)),
    CalibrationPoint("u3", True, sig(0.85, 0.1)),
]


def test_calibration_picks_the_pair_that_catches_most_within_the_false_refusal_bound() -> None:
    cal = calibrate(POINTS, max_false_refusal_pct=25)  # 4 answerable: at most 1 false refusal
    # cosine candidates: -0.45 0.575 0.625 0.675 0.75 0.825 0.875 1.9 (8); BM25 candidates:
    # -0.9 0.15 0.35 0.75 2 3.5 4.5 6 (8); so the grid has 64 pairs
    assert len(cal.grid) == 64
    # To catch all three, tau_cos must pass u3's 0.85 (so 0.875) and tau_bm25 must pass u1's 0.5
    # (so at least 0.75). With (0.875, 0.75) a2, a3, a4 have BM25 4, 3, 1, all >= 0.75: no false
    # refusal. Lower thresholds that also catch all three do not exist, so this is the tie-break.
    best = cal.cosine_bm25
    assert (best.thresholds.cosine, best.thresholds.bm25) == (0.875, 0.75)
    assert (best.metrics.caught, best.metrics.false_refusals) == (3, 0)


def test_cosine_only_uses_the_always_true_bm25_axis_and_the_same_bound() -> None:
    cal = calibrate(POINTS, max_false_refusal_pct=25)
    only = cal.cosine_only
    # BM25 axis always true = max 5 + 1 = 6. Catching u3 needs tau_cos 0.875, which also refuses
    # a2, a3, a4 (3 false refusals > 1). tau_cos 0.675 catches u1, u2 and refuses only a4:
    # 2 caught, 1 false refusal, feasible; nothing feasible catches more.
    assert (only.thresholds.cosine, only.thresholds.bm25) == (0.675, 6.0)
    assert (only.metrics.caught, only.metrics.false_refusals) == (2, 1)


def test_a_tighter_bound_gives_a_different_cosine_only_choice() -> None:
    only = calibrate(POINTS, max_false_refusal_pct=0).cosine_only
    # no false refusal allowed: tau_cos 0.575 catches only u1 (0.55) and refuses no answerable
    assert (only.thresholds.cosine, only.metrics.caught, only.metrics.false_refusals) == (
        0.575, 1, 0
    )  # fmt: skip


def test_the_bound_is_checked_in_integers() -> None:
    # 5% of 54 answerable queries is 2.7, so 2 false refusals pass and 3 do not (3 * 100 > 5 * 54)
    rows_answerable = 54
    assert 5 * rows_answerable >= 2 * 100
    assert not 5 * rows_answerable >= 3 * 100


def test_ties_go_to_the_lowest_cosine_then_the_lowest_bm25() -> None:
    # one unanswerable (0.5, 0.5) and one answerable (0.9, 5): many pairs catch the unanswerable
    # with no false refusal; the lowest tau_cos that does is the midpoint 0.7 and the lowest
    # tau_bm25 is 2.75
    points = [CalibrationPoint("a", False, sig(0.9, 5)), CalibrationPoint("u", True, sig(0.5, 0.5))]
    best = calibrate(points, 5).cosine_bm25
    # cosine candidates for distinct 0.5, 0.9: [-0.5, 0.7, 1.9]; BM25 for 0.5, 5: [-0.5, 2.75, 6]
    assert (best.thresholds.cosine, best.thresholds.bm25) == (0.7, 2.75)


def test_calibration_needs_both_kinds_of_query() -> None:
    with pytest.raises(CalibrationError):
        calibrate([CalibrationPoint("a", False, sig(0.9, 5))], 5)
    with pytest.raises(CalibrationError):
        calibrate([CalibrationPoint("u", True, sig(0.5, 1))], 5)


def test_metrics_at_applies_the_stage_one_rule() -> None:
    m = metrics_at(POINTS, Thresholds(cosine=0.875, bm25=0.75))
    assert (m.caught, m.false_refusals, m.answerable, m.unanswerable) == (3, 0, 4, 3)
