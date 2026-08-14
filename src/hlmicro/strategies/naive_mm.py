"""Naive symmetric market maker: fixed spread around microprice, fixed
size, hard inventory cap. The baseline every fancier strategy should beat."""

from __future__ import annotations

from hlmicro.strategies.base import Fill, MarketState, Quote, Strategy


class NaiveSymmetricMM(Strategy):
    name = "naive_mm"

    def __init__(self, half_spread_bps: float, quote_size: float, max_inventory: float) -> None:
        self.half_spread_bps = half_spread_bps
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.inventory: float = 0.0

    def on_book_update(self, state: MarketState) -> None:
        pass  # stateless beyond inventory, which on_fill maintains

    def on_fill(self, fill: Fill) -> None:
        self.inventory += fill.signed_size

    def get_quotes(self, state: MarketState) -> Quote:
        if state.liquidity_drought:
            return Quote(bid_px=None, bid_sz=0.0, ask_px=None, ask_sz=0.0)

        half = state.microprice * self.half_spread_bps / 10_000
        bid_px = state.microprice - half
        ask_px = state.microprice + half

        bid_sz = self.quote_size if self.inventory < self.max_inventory else 0.0
        ask_sz = self.quote_size if self.inventory > -self.max_inventory else 0.0
        return Quote(bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz)
