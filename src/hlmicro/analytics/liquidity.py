"""Liquidity-change detection: rolling total depth within X bps of mid,
flagged for statistically unusual drops via a rolling z-score.

This is a simple, transparent changepoint heuristic (not a full CUSUM/
Bayesian changepoint model) - defensible for a portfolio project and easy
to reason about, at the cost of being slower to react than more elaborate
methods. That tradeoff is intentional and documented rather than hidden.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def depth_within_bps(
    bid_px: list[float],
    bid_sz: list[float],
    ask_px: list[float],
    ask_sz: list[float],
    mid: float,
    bps_window: float,
) -> float:
    if mid == 0:
        return 0.0
    threshold = mid * bps_window / 10_000
    bid_depth = sum(sz for px, sz in zip(bid_px, bid_sz, strict=True) if mid - px <= threshold)
    ask_depth = sum(sz for px, sz in zip(ask_px, ask_sz, strict=True) if px - mid <= threshold)
    return bid_depth + ask_depth


def compute_depth_within_bps_batch(df: pl.DataFrame, bps_window: float) -> pl.DataFrame:
    """Requires a `mid` column (see spread.compute_spread_batch)."""
    bid_px = np.array(df["bid_px"].to_list(), dtype=np.float64)
    bid_sz = np.array(df["bid_sz"].to_list(), dtype=np.float64)
    ask_px = np.array(df["ask_px"].to_list(), dtype=np.float64)
    ask_sz = np.array(df["ask_sz"].to_list(), dtype=np.float64)
    mid = df["mid"].to_numpy()[:, None]  # broadcast over levels

    threshold = mid * bps_window / 10_000
    bid_mask = (mid - bid_px) <= threshold
    ask_mask = (ask_px - mid) <= threshold
    depth = (bid_sz * bid_mask).sum(axis=1) + (ask_sz * ask_mask).sum(axis=1)

    return df.with_columns(pl.Series(f"depth_within_{int(bps_window)}bps", depth))


def rolling_liquidity_zscore(depth: pl.Series, window: int) -> pl.Series:
    roll_mean = depth.rolling_mean(window_size=window, min_samples=max(2, window // 4))
    roll_std = depth.rolling_std(window_size=window, min_samples=max(2, window // 4))
    return (depth - roll_mean) / roll_std


def flag_liquidity_drought(z_scores: pl.Series, threshold: float = -2.5) -> pl.Series:
    """True where depth has dropped statistically unusually far below its
    recent rolling mean (z-score below threshold). NaN z-scores (not
    enough history yet) are treated as "no signal", not a drought."""
    return z_scores.fill_nan(0.0) < threshold
