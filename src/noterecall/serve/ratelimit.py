"""The per-IP daily rate limit and the client-IP rule. Implements proposal 009.

`RateLimiter` allows `limit` questions per client IP per UTC day, counted in memory (it starts
again when the service restarts, and it is per instance). A question is counted when it is
accepted, before any model call, so that failures cannot be used to spend more. Counters of
earlier days are dropped when the day changes.

`resolve_client_ip` decides which IP a request is charged to. The UI sends the end user's IP in a
header together with a shared secret; the API believes that header only when the secret matches
(compared in constant time). Otherwise the request is charged to its own caller: the first
`X-Forwarded-For` entry (Render's proxy puts the connecting client there), or the socket peer when
there is no such header. Every candidate is parsed as an IP address, so no arbitrary string
becomes a counter key."""

import hmac
import ipaddress
import threading
from datetime import UTC, date, datetime, timedelta
from typing import NamedTuple

from noterecall.srs.clock import Clock

UNKNOWN_IP = "unknown"


class Decision(NamedTuple):
    """Whether a question may be asked, how many are left today, and when the day ends."""

    allowed: bool
    remaining: int
    retry_after_s: int


def parse_ip(value: str | None) -> str | None:
    """The normalized form of the first address in a header value, or None if it is not an IP."""
    if not value:
        return None
    first = value.split(",")[0].strip()
    try:
        return str(ipaddress.ip_address(first))
    except ValueError:
        return None


def resolve_client_ip(
    forwarded_for: str | None,
    ui_client_ip: str | None,
    ui_secret: str | None,
    expected_secret: str | None,
    peer: str | None,
) -> str:
    """The IP a request is charged to (see the module docstring)."""
    if expected_secret and ui_secret is not None:
        matches = hmac.compare_digest(ui_secret.encode(), expected_secret.encode())
        forwarded = parse_ip(ui_client_ip) if matches else None
        if forwarded is not None:
            return forwarded
    return parse_ip(forwarded_for) or parse_ip(peer) or UNKNOWN_IP


class RateLimiter:
    """At most `limit` accepted questions per IP per UTC day."""

    def __init__(self, limit: int, clock: Clock) -> None:
        self._limit = limit
        self._clock = clock
        self._lock = threading.Lock()
        self._day: date | None = None
        self._counts: dict[str, int] = {}

    def check(self, ip: str) -> Decision:
        """Count a question for `ip` if it is within the limit."""
        now = self._clock.now().astimezone(UTC)
        with self._lock:
            if self._day != now.date():
                self._day, self._counts = now.date(), {}
            used = self._counts.get(ip, 0)
            if used >= self._limit:
                return Decision(False, 0, seconds_to_midnight(now))
            self._counts[ip] = used + 1
            return Decision(True, self._limit - used - 1, seconds_to_midnight(now))

    def tracked(self) -> int:
        """How many IPs have a counter today (for tests and the health check)."""
        with self._lock:
            return len(self._counts)


def seconds_to_midnight(now: datetime) -> int:
    """Whole seconds until the next 00:00 UTC (at least 1)."""
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))
