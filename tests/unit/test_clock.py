"""The injectable clock."""

from datetime import UTC, datetime, timedelta

from noterecall.srs.clock import FixedClock, SystemClock


def test_system_clock_is_aware_utc() -> None:
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_fixed_clock_moves_only_when_advanced() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    clock = FixedClock(start)
    assert clock.now() == clock.now() == start
    clock.advance(timedelta(days=1))
    assert clock.now() == datetime(2026, 1, 2, tzinfo=UTC)
