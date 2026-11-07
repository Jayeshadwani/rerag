"""The per-IP daily limit and the client-IP rule (proposal 009)."""

from datetime import UTC, datetime, timedelta

import pytest

from noterecall.serve.ratelimit import (
    UNKNOWN_IP,
    RateLimiter,
    parse_ip,
    resolve_client_ip,
    seconds_to_midnight,
)
from noterecall.srs.clock import FixedClock

SECRET = "s3cret"
START = datetime(2026, 3, 1, 23, 0, 0, tzinfo=UTC)


def test_parse_ip_takes_the_first_valid_address() -> None:
    assert parse_ip("203.0.113.5, 10.0.0.1") == "203.0.113.5"
    assert parse_ip("2001:db8::1") == "2001:db8::1"
    assert parse_ip("not-an-ip") is None
    assert parse_ip("") is None
    assert parse_ip(None) is None


def test_valid_secret_makes_the_forwarded_ip_count() -> None:
    ip = resolve_client_ip("198.51.100.1", "203.0.113.9", SECRET, SECRET, "10.0.0.2")
    assert ip == "203.0.113.9"


def test_spoof_without_secret_is_charged_to_the_caller() -> None:
    ip = resolve_client_ip("198.51.100.1", "203.0.113.9", None, SECRET, "10.0.0.2")
    assert ip == "198.51.100.1"


def test_spoof_with_wrong_secret_is_charged_to_the_caller() -> None:
    ip = resolve_client_ip("198.51.100.1, 10.0.0.1", "203.0.113.9", "wrong", SECRET, "10.0.0.2")
    assert ip == "198.51.100.1"


def test_no_server_secret_never_trusts_the_header() -> None:
    assert resolve_client_ip(None, "203.0.113.9", "", None, "10.0.0.2") == "10.0.0.2"
    assert resolve_client_ip(None, "203.0.113.9", "", "", "10.0.0.2") == "10.0.0.2"


def test_a_garbage_forwarded_ip_falls_back_to_the_caller() -> None:
    assert resolve_client_ip("198.51.100.1", "garbage", SECRET, SECRET, None) == "198.51.100.1"
    assert resolve_client_ip(None, None, None, SECRET, None) == UNKNOWN_IP


def test_the_sixteenth_question_is_blocked() -> None:
    limiter = RateLimiter(15, FixedClock(START))
    decisions = [limiter.check("1.1.1.1") for _ in range(16)]
    assert [d.allowed for d in decisions] == [True] * 15 + [False]
    assert decisions[0].remaining == 14
    assert decisions[14].remaining == 0
    assert decisions[15].remaining == 0
    assert limiter.check("2.2.2.2").allowed  # another IP has its own count
    assert limiter.tracked() == 2


def test_the_count_starts_again_on_the_next_utc_day() -> None:
    clock = FixedClock(START)
    limiter = RateLimiter(2, clock)
    limiter.check("1.1.1.1")
    limiter.check("1.1.1.1")
    assert not limiter.check("1.1.1.1").allowed
    clock.advance(timedelta(hours=1))  # 00:00 UTC of the next day
    assert limiter.check("1.1.1.1").allowed
    assert limiter.tracked() == 1  # the old day's counters were dropped


def test_retry_after_is_the_time_to_midnight() -> None:
    limiter = RateLimiter(1, FixedClock(START))
    limiter.check("1.1.1.1")
    assert limiter.check("1.1.1.1").retry_after_s == 3600  # 23:00 -> 00:00


def test_seconds_to_midnight_is_at_least_one() -> None:
    assert seconds_to_midnight(datetime(2026, 3, 1, 23, 59, 59, 999999, tzinfo=UTC)) == 1
    assert seconds_to_midnight(datetime(2026, 3, 1, 0, 0, 0, tzinfo=UTC)) == 86400


def test_fixed_clock_needs_utc() -> None:
    with pytest.raises(ValueError, match="UTC"):
        FixedClock(datetime(2026, 3, 1))
    assert FixedClock(START).now() == START
