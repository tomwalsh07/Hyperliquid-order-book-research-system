"""Inventory-aware market maker, adapted from Avellaneda-Stoikov (2008).

The original model quotes around a reservation price that skews against
inventory, shrinking as a fixed terminal time T approaches:

    r = s - q * gamma * sigma^2 * (T - t)
    delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

Two adaptations for a perpetual (no terminal time, and no robust way to
calibrate k, the order-arrival-intensity parameter, from L2-only data):

1. **(T - t) -> a fixed rolling `decay_window_s`.** Instead of urgency
   rising as a session-close approaches (which doesn't exist here), we use
   a constant "effective horizon" - the strategy always behaves as if it
   cares about risk over roughly the next `decay_window_s` seconds. This
   is a receding-horizon simplification, not the textbook model, and is
   documented as such rather than silently repurposing (T-t)->0.

2. **The `(2/gamma)*ln(1+gamma/k)` spread-widening term is dropped.**
   Calibrating k well needs an order-arrival-intensity model fit to trade
   data at multiple price offsets, which is a project in itself and not
   reliably estimable from the data volume this system collects. In its
   place we use a configurable `base_half_spread_bps` floor plus a
   volatility-scaled term, so the strategy still widens when the market is
   choppy, just via a simpler, honestly-labeled heuristic instead of a
   mis-calibrated version of the "real" formula.

Volatility (`sigma`) is estimated as the standard deviation of consecutive
mid-price *differences* (not log returns) over `vol_lookback_s` - this
keeps sigma in price units so `sigma**2 * decay_window_s` combines with
`risk_aversion` to produce a price-unit skew directly, without an implicit
unit-conversion step. It is a per-update (not per-second) estimate; given
irregular update spacing this is an approximation, not a proper diffusion
coefficient - see docs/methodology.md.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from hlmicro.strategies.base import Fill, MarketState, Quote, Strategy


class InventoryAwareMM(Strategy):
    name = "inventory_mm"

    def __init__(
        self,
        risk_aversion: float,
        vol_lookback_s: float,
        decay_window_s: float,
        max_inventory: float,
        quote_size: float,
        base_half_spread_bps: float = 5.0,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.vol_lookback_s = vol_lookback_s
        self.decay_window_s = decay_window_s
        self.max_inventory = max_inventory
        self.quote_size = quote_size
        self.base_half_spread_bps = base_half_spread_bps

        self.inventory: float = 0.0
        self._mid_history: deque[tuple[int, float]] = deque()

    def on_book_update(self, state: MarketState) -> None:
        self._mid_history.append((state.timestamp_ms, state.mid))
        cutoff = state.timestamp_ms - self.vol_lookback_s * 1000
        while self._mid_history and self._mid_history[0][0] < cutoff:
            self._mid_history.popleft()

    def on_fill(self, fill: Fill) -> None:
        self.inventory += fill.signed_size

    def _price_volatility(self) -> float:
        if len(self._mid_history) < 3:
            return 0.0
        prices = np.array([p for _, p in self._mid_history])
        diffs = np.diff(prices)
        return float(np.std(diffs))

    def get_quotes(self, state: MarketState) -> Quote:
        if state.liquidity_drought:
            return Quote(bid_px=None, bid_sz=0.0, ask_px=None, ask_sz=0.0)

        sigma = self._price_volatility()
        risk_term = self.risk_aversion * sigma**2 * self.decay_window_s

        reservation_price = state.mid - self.inventory * risk_term
        half_spread = max(
            self.base_half_spread_bps / 10_000 * state.mid,
            0.5 * risk_term,
        )

        bid_px = reservation_price - half_spread
        ask_px = reservation_price + half_spread

        bid_sz = self.quote_size if self.inventory < self.max_inventory else 0.0
        ask_sz = self.quote_size if self.inventory > -self.max_inventory else 0.0
        return Quote(bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz)
