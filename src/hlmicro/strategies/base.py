"""Strategy interface shared by all market-making strategies.

The backtester (and a live executor, if one is ever added) drives
a strategy purely through this interface: feed it market state, ask for
quotes, tell it about fills. A strategy owns its own inventory and risk
bookkeeping; the engine owns fills/PnL/fees, which it computes independently
so a strategy can't accidentally mark its own homework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

FillSide = Literal["buy", "sell"]


@dataclass
class MarketState:
    coin: str
    timestamp_ms: int
    best_bid: float
    best_bid_sz: float
    best_ask: float
    best_ask_sz: float
    mid: float
    microprice: float
    tob_imbalance: float
    liquidity_drought: bool = False


@dataclass
class Quote:
    """A desired resting order per side. size == 0 means "don't quote this
    side right now" (risk limit breached, or a kill-switch is active)."""

    bid_px: float | None
    bid_sz: float
    ask_px: float | None
    ask_sz: float


@dataclass
class Fill:
    timestamp_ms: int
    side: FillSide
    price: float
    size: float

    @property
    def signed_size(self) -> float:
        return self.size if self.side == "buy" else -self.size


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def on_book_update(self, state: MarketState) -> None:
        """Called on every book snapshot, before get_quotes. Used to update
        internal state (volatility estimates, etc.) - must not have side
        effects the backtester depends on for fills/PnL."""

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        """Called by the engine after it decides a fill happened. The
        strategy updates its own inventory here - it is the single source
        of truth for "how much do I currently hold"."""

    @abstractmethod
    def get_quotes(self, state: MarketState) -> Quote:
        """Desired resting bid/ask given current state and this strategy's
        own inventory. Must respect its own configured risk limits."""
