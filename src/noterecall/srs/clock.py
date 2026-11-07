"""The injectable clock (CLAUDE.md: time goes through here; tests use a fixed UTC clock).

`SystemClock` is the real time, always timezone-aware UTC. `FixedClock` holds a time that tests
set and advance. The rate limiter of the service reads the UTC date from a `Clock`."""

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    """Anything that can say what time it is, in UTC."""

    def now(self) -> datetime: ...


class SystemClock:
    """The real clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """A clock that only moves when told to."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None or start.utcoffset() != timedelta(0):
            raise ValueError("a clock time must be timezone-aware UTC")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta
