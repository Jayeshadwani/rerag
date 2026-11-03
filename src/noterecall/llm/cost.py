"""Cost from usage, and the spend cap. Implements the owner's `max_usd` rule (proposal 004).

Cost is computed from the usage the provider reports and the owner-supplied prices (USD per one
million tokens): ordinary input at the input price, cached input at the cached price, and output,
which includes reasoning tokens, at the output price. Tokens written to the cache count as
ordinary input.
The meter starts from what the cache ledger says was already spent, so the cap is a cap on the whole
milestone and not on one run. It stops new calls once the spend reaches the cap; calls already in
flight may overshoot by at most the cost of one call per worker.
"""

import threading
from collections.abc import Iterable, Mapping

from noterecall.config import PriceConfig
from noterecall.llm.cache import SpendRow
from noterecall.llm.client import BudgetExceededError, Usage

PER = 1_000_000


def cost_usd(usage: Usage, price: PriceConfig) -> float:
    """Dollars for one usage record."""
    fresh = usage.input_tokens - usage.cached_tokens
    return (
        fresh * price.input
        + usage.cached_tokens * price.cached_input
        + usage.output_tokens * price.output
    ) / PER


def ledger_cost(rows: Iterable[SpendRow], prices: Mapping[str, PriceConfig]) -> float:
    """Dollars for everything in the spend ledger."""
    return sum(cost_usd(row.usage, prices[row.model]) for row in rows)


class CostMeter:
    """Tracks the money spent and refuses new calls at the cap."""

    def __init__(
        self, prices: Mapping[str, PriceConfig], cap_usd: float, already_spent_usd: float = 0.0
    ) -> None:
        self._prices = prices
        self._cap = cap_usd
        self._spent = already_spent_usd
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    def check(self) -> None:
        """Raise if the cap has been reached."""
        with self._lock:
            if self._spent >= self._cap:
                raise BudgetExceededError(
                    f"spent ${self._spent:.4f} of the ${self._cap:.2f} cap; no further calls"
                )

    def add(self, model: str, usage: Usage) -> float:
        """Record a finished call and return its cost."""
        cost = cost_usd(usage, self._prices[model])
        with self._lock:
            self._spent += cost
        return cost
