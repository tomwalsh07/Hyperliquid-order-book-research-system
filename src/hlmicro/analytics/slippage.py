"""Slippage estimation: walk the book from best price outward, compute the
size-weighted average execution price, report the cost vs. mid in bps.

This is a *pre-trade* estimate against a resting-liquidity snapshot, not a
fill simulation - it assumes you can execute the full visible depth
instantly at the displayed sizes, which is optimistic (ignores latency,
other participants racing the same liquidity, and any size beyond the
captured depth). See backtest/fills.py for the separate, more conservative
treatment used when actually crediting simulated fills.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

Side = Literal["buy", "sell"]


def walk_the_book(levels: list[tuple[float, float]], order_size: float) -> tuple[float, float]:
    """levels: [(price, size), ...] sorted best-to-worst (as returned by
    OrderBook.bid_levels()/.ask_levels()). Returns (vwap_price, filled_size);
    filled_size < order_size if the visible depth runs out."""
    remaining = order_size
    notional = 0.0
    filled = 0.0
    for px, sz in levels:
        take = min(remaining, sz)
        notional += take * px
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled == 0:
        return float("nan"), 0.0
    return notional / filled, filled


def slippage_bps(vwap: float, mid: float, side: Side) -> float:
    if mid == 0:
        return float("nan")
    if side == "buy":
        return (vwap - mid) / mid * 10_000
    return (mid - vwap) / mid * 10_000


def estimate_slippage(
    levels: list[tuple[float, float]], order_size: float, mid: float, side: Side
) -> dict:
    vwap, filled = walk_the_book(levels, order_size)
    return {
        "side": side,
        "order_size": order_size,
        "filled_size": filled,
        "fully_filled": filled >= order_size,
        "vwap": vwap,
        "mid": mid,
        "slippage_bps": slippage_bps(vwap, mid, side) if filled > 0 else float("nan"),
    }


def compute_slippage_batch(df: pl.DataFrame, order_sizes: list[float], side: Side) -> pl.DataFrame:
    """Row-by-row walk-the-book over a stored l2book snapshot table. Not a
    hot-path vectorized operation (the walk is inherently sequential per
    row) - fine for a slippage report over the collected dataset, but not
    intended to run per-row inside the high-frequency research pipeline."""
    side_col = "ask" if side == "buy" else "bid"
    px_col, sz_col = f"{side_col}_px", f"{side_col}_sz"

    px_lists = df[px_col].to_list()
    sz_lists = df[sz_col].to_list()
    mids = df["mid"].to_list()

    out_cols: dict[str, list[float]] = {f"slippage_bps_{int(s)}": [] for s in order_sizes}
    for size in order_sizes:
        col = out_cols[f"slippage_bps_{int(size)}"]
        for px_row, sz_row, mid in zip(px_lists, sz_lists, mids, strict=True):
            levels = list(zip(px_row, sz_row, strict=True))
            vwap, filled = walk_the_book(levels, size)
            col.append(slippage_bps(vwap, mid, side) if filled > 0 else float("nan"))

    return df.with_columns(**{k: pl.Series(k, v) for k, v in out_cols.items()})
