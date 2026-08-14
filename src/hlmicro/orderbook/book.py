"""In-memory limit order book, for the live/streaming use case.

Batch analytics over stored history do NOT go through this class — every
stored `l2book` row already is a complete snapshot (see docs/api_notes.md
§1), so batch code operates directly on the Parquet columns with vectorized
polars/numpy. This class exists for (a) a live strategy's `on_book_update`
hook and (b) exercising update semantics (upsert/delete/crossed-book) in
isolation under test, independent of whatever transport delivered them.

`load_snapshot` is the path real Hyperliquid traffic exercises today.
`apply_level_update` (single-price upsert-or-delete) is a lower-level
primitive kept for testability and so this class could serve a genuinely
incremental feed (e.g. l4Book) later without a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Side = Literal["bid", "ask"]


@dataclass
class OrderBook:
    symbol: str
    depth_levels: int = 20
    bids: dict[float, float] = field(default_factory=dict)  # price -> size
    asks: dict[float, float] = field(default_factory=dict)
    bid_n: dict[float, int] = field(default_factory=dict)  # price -> resting order count
    ask_n: dict[float, int] = field(default_factory=dict)
    last_update_ms: int | None = None

    def load_snapshot(
        self,
        bid_px: list[float],
        bid_sz: list[float],
        ask_px: list[float],
        ask_sz: list[float],
        exch_time_ms: int,
        bid_n: list[int] | None = None,
        ask_n: list[int] | None = None,
    ) -> None:
        """Full replace — the only path live Hyperliquid l2Book traffic uses."""
        self.bids = dict(zip(bid_px, bid_sz, strict=True))
        self.asks = dict(zip(ask_px, ask_sz, strict=True))
        self.bid_n = dict(zip(bid_px, bid_n, strict=True)) if bid_n else {}
        self.ask_n = dict(zip(ask_px, ask_n, strict=True)) if ask_n else {}
        self.last_update_ms = exch_time_ms

    def apply_level_update(
        self, side: Side, price: float, size: float, n: int | None = None
    ) -> None:
        """Upsert a level, or delete it if size == 0. Not exercised by live
        Hyperliquid l2Book traffic (which is snapshot-only) but kept for
        testability and future diff-based feeds."""
        book = self.bids if side == "bid" else self.asks
        counts = self.bid_n if side == "bid" else self.ask_n
        if size == 0:
            book.pop(price, None)
            counts.pop(price, None)
        else:
            book[price] = size
            if n is not None:
                counts[price] = n

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.bid_n.clear()
        self.ask_n.clear()
        self.last_update_ms = None

    def best_bid(self) -> tuple[float, float] | None:
        if not self.bids:
            return None
        px = max(self.bids)
        return px, self.bids[px]

    def best_ask(self) -> tuple[float, float] | None:
        if not self.asks:
            return None
        px = min(self.asks)
        return px, self.asks[px]

    def is_crossed(self) -> bool:
        """True if best_bid >= best_ask, which should never happen on a
        healthy book — flags a data/reconstruction bug if it does."""
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return False
        return bb[0] >= ba[0]

    def bid_levels(self, n: int | None = None) -> list[tuple[float, float]]:
        """Descending by price (best bid first)."""
        n = n if n is not None else self.depth_levels
        return sorted(self.bids.items(), key=lambda kv: -kv[0])[:n]

    def ask_levels(self, n: int | None = None) -> list[tuple[float, float]]:
        """Ascending by price (best ask first)."""
        n = n if n is not None else self.depth_levels
        return sorted(self.asks.items(), key=lambda kv: kv[0])[:n]

    def mid_price(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb[0] + ba[0]) / 2
