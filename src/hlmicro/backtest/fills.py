"""Fill simulation heuristic for an L2-only backtest.

**The core limitation, stated up front**: L2 book data has no queue
position. We can see that N units are resting at a price level, but not
where in the queue our hypothetical order would sit relative to everyone
else's. Any L2-only backtest that claims to know exactly when a resting
order fills is overstating its precision.

**The heuristic used here**: we credit a fill against our resting quote at
price P only once *cumulative* trade volume printed at P (matching our
side) reaches `min_consumption_pct` (default 50%) of the size that was
resting at P when we last placed/moved our quote there. This is
deliberately conservative - a naive "any print at my price = I'm filled"
model over-credits fills (real queue position means you often *don't* get
filled by a print that merely ties your price), and this heuristic assumes
we sit roughly in the back half of the queue at a price level, which is a
reasonable prior for a passive quote arriving after the level was already
built up. It is still just a heuristic, not ground truth - documented
prominently here and in docs/methodology.md / the README, not hidden.

Once the threshold is crossed, we credit a fill for min(remaining quote
size, the trade volume that crossed us over the threshold) and reset the
consumption counter for that (side, price) - as if we re-queue at the back
after being filled, consistent with not tracking true queue position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Side = Literal["bid", "ask"]
TradeSide = Literal["B", "A"]  # B = taker bought (hits ask), A = taker sold (hits bid)

_HITS = {"bid": "A", "ask": "B"}  # our resting side -> the trade side that can hit it


@dataclass
class FillSimulator:
    min_consumption_pct: float = 0.5

    _resting_depth: dict[Side, float] = field(default_factory=dict)
    _resting_price: dict[Side, float | None] = field(
        default_factory=lambda: {"bid": None, "ask": None}
    )
    _consumed: dict[Side, float] = field(default_factory=lambda: {"bid": 0.0, "ask": 0.0})

    def set_quote(
        self, side: Side, price: float | None, resting_depth_at_price: float | None
    ) -> None:
        """Call whenever the strategy's quote for `side` changes price (or
        is withdrawn, price=None). Moving to a new price resets the
        consumption counter - we have no history of trades at a price we
        just started quoting. `resting_depth_at_price` is the book depth
        at that price *excluding* our own hypothetical order (we're not
        really in the book), taken from the most recent l2book snapshot."""
        if price != self._resting_price[side]:
            self._consumed[side] = 0.0
        self._resting_price[side] = price
        self._resting_depth[side] = (
            resting_depth_at_price if resting_depth_at_price is not None else 0.0
        )

    def resting_price(self, side: Side) -> float | None:
        return self._resting_price[side]

    def on_trade(
        self, trade_side: TradeSide, trade_price: float, trade_size: float
    ) -> dict[Side, float]:
        """Feed one trade print. Returns {side: fill_size} for any side(s)
        this trade fills (usually at most one, since bid/ask are at
        different prices, but both are checked independently)."""
        fills: dict[Side, float] = {}
        for side in ("bid", "ask"):
            price = self._resting_price[side]
            if price is None or trade_side != _HITS[side] or trade_price != price:
                continue
            depth = self._resting_depth[side]
            if depth <= 0:
                # No visibility into resting depth at this price (outside
                # captured book depth) -> conservative choice: no fill,
                # not an optimistic "assume thin book, fill immediately".
                continue
            threshold = self.min_consumption_pct * depth
            new_consumed = self._consumed[side] + trade_size
            if new_consumed >= threshold:
                fill_size = new_consumed - threshold
                if fill_size > 0:
                    fills[side] = fill_size
                self._consumed[side] = 0.0  # re-queue at the back after a credited fill
            else:
                self._consumed[side] = new_consumed
        return fills
