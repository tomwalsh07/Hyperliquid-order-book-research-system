"""Bid/ask spread: absolute and basis points."""

from __future__ import annotations

import polars as pl


def spread_abs(best_bid: float, best_ask: float) -> float:
    return best_ask - best_bid


def spread_bps(best_bid: float, best_ask: float) -> float:
    mid = (best_bid + best_ask) / 2
    if mid == 0:
        return float("nan")
    return (best_ask - best_bid) / mid * 10_000


def mid_price(best_bid: float, best_ask: float) -> float:
    return (best_bid + best_ask) / 2


def compute_spread_batch(df: pl.DataFrame) -> pl.DataFrame:
    """Adds best_bid, best_ask, mid, spread_abs, spread_bps to an l2book
    DataFrame (schema: bid_px/ask_px as List(Float64), best level first)."""
    best_bid = pl.col("bid_px").list.get(0)
    best_ask = pl.col("ask_px").list.get(0)
    mid = (best_bid + best_ask) / 2
    return df.with_columns(
        best_bid.alias("best_bid"),
        best_ask.alias("best_ask"),
        mid.alias("mid"),
        (best_ask - best_bid).alias("spread_abs"),
        ((best_ask - best_bid) / mid * 10_000).alias("spread_bps"),
    )
