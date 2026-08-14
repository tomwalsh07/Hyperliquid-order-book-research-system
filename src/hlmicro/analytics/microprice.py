"""Microprice (Stoikov): a size-weighted mid that leans toward whichever
side has less resting size, since that's the side more likely to be
walked through next.

    microprice = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
"""

from __future__ import annotations

import polars as pl


def microprice(best_bid: float, best_bid_sz: float, best_ask: float, best_ask_sz: float) -> float:
    denom = best_bid_sz + best_ask_sz
    if denom == 0:
        return (best_bid + best_ask) / 2
    return (best_bid * best_ask_sz + best_ask * best_bid_sz) / denom


def microprice_mid_deviation_bps(microprice_val: float, mid: float) -> float:
    """How far microprice has drifted from the plain mid, in bps. A
    feature for the research module: large deviation = size
    imbalance is already pricing in a directional lean."""
    if mid == 0:
        return float("nan")
    return (microprice_val - mid) / mid * 10_000


def compute_microprice_batch(df: pl.DataFrame) -> pl.DataFrame:
    """Requires best_bid/best_ask/mid columns (see spread.compute_spread_batch)."""
    best_bid_sz = pl.col("bid_sz").list.get(0)
    best_ask_sz = pl.col("ask_sz").list.get(0)
    denom = best_bid_sz + best_ask_sz
    micro = (
        pl.when(denom == 0)
        .then(pl.col("mid"))
        .otherwise((pl.col("best_bid") * best_ask_sz + pl.col("best_ask") * best_bid_sz) / denom)
    )
    return df.with_columns(micro.alias("microprice")).with_columns(
        ((pl.col("microprice") - pl.col("mid")) / pl.col("mid") * 10_000).alias(
            "microprice_mid_dev_bps"
        )
    )
