"""Order-book imbalance: top-of-book and depth-weighted.

    I = (bid_size - ask_size) / (bid_size + ask_size)   in [-1, 1]

Positive I = more resting size on the bid = (weak) buy-side pressure.
"""

from __future__ import annotations

import numpy as np
import polars as pl

LevelWeighting = str  # "linear_decay" | "uniform" | "exp_decay"


def top_of_book_imbalance(bid_sz: float, ask_sz: float) -> float:
    denom = bid_sz + ask_sz
    if denom == 0:
        return 0.0
    return (bid_sz - ask_sz) / denom


def level_weights(n: int, scheme: LevelWeighting = "linear_decay") -> np.ndarray:
    """Weight applied to each depth level, index 0 = best price.

    linear_decay: weight n, n-1, ..., 1 - closest levels dominate but
        deeper levels still count a little. Linear rather than
        exponential decay keeps level 20 non-negligible
        (~1/n of level 0's weight) rather than ~0, which matters when
        book depth is thin.
    uniform: every level in [0, n) counts equally - a coarser, simpler
        alternative kept for comparison in the research module.
    exp_decay: weight halves each level out (0.5**i) - the aggressive
        end of the spectrum, kept for sensitivity checks.
    """
    if scheme == "linear_decay":
        return np.arange(n, 0, -1, dtype=np.float64)
    if scheme == "uniform":
        return np.ones(n, dtype=np.float64)
    if scheme == "exp_decay":
        return 0.5 ** np.arange(n, dtype=np.float64)
    raise ValueError(f"Unknown level weighting scheme: {scheme}")


def depth_weighted_imbalance(
    bid_sizes: list[float], ask_sizes: list[float], n: int, scheme: LevelWeighting = "linear_decay"
) -> float:
    weights = level_weights(n, scheme)
    bid_arr = np.asarray(bid_sizes[:n], dtype=np.float64)
    ask_arr = np.asarray(ask_sizes[:n], dtype=np.float64)
    # pad short books (near listing, thin symbols) with zeros rather than error
    if len(bid_arr) < n:
        bid_arr = np.pad(bid_arr, (0, n - len(bid_arr)))
    if len(ask_arr) < n:
        ask_arr = np.pad(ask_arr, (0, n - len(ask_arr)))

    bid_depth = float(np.dot(weights, bid_arr))
    ask_depth = float(np.dot(weights, ask_arr))
    denom = bid_depth + ask_depth
    if denom == 0:
        return 0.0
    return (bid_depth - ask_depth) / denom


def compute_imbalance_batch(
    df: pl.DataFrame,
    depth_levels: tuple[int, ...] = (5, 10, 20),
    scheme: LevelWeighting = "linear_decay",
) -> pl.DataFrame:
    """Adds tob_imbalance and imbalance_d{N} for each N in depth_levels."""
    best_bid_sz = pl.col("bid_sz").list.get(0)
    best_ask_sz = pl.col("ask_sz").list.get(0)
    tob_denom = best_bid_sz + best_ask_sz
    out = df.with_columns(
        pl.when(tob_denom == 0)
        .then(0.0)
        .otherwise((best_bid_sz - best_ask_sz) / tob_denom)
        .alias("tob_imbalance")
    )

    bid_mat_full = np.array(df["bid_sz"].to_list(), dtype=np.float64)
    ask_mat_full = np.array(df["ask_sz"].to_list(), dtype=np.float64)
    max_n = bid_mat_full.shape[1]

    for n in depth_levels:
        eff_n = min(n, max_n)
        weights = level_weights(eff_n, scheme)
        bid_depth = bid_mat_full[:, :eff_n] @ weights
        ask_depth = ask_mat_full[:, :eff_n] @ weights
        denom = bid_depth + ask_depth
        imb = np.divide(bid_depth - ask_depth, denom, out=np.zeros_like(denom), where=denom != 0)
        out = out.with_columns(pl.Series(f"imbalance_d{n}", imb))

    return out
