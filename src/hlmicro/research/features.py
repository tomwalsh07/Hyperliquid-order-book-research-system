"""Feature construction for the imbalance -> forward-return study.

Every function here is causal: each row's feature value uses only data
timestamped at or before that row's own timestamp. Forward-looking labels
live in labels.py, kept deliberately separate so there is exactly one place
in the codebase where "look at the future" is allowed to happen.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hlmicro.analytics.imbalance import compute_imbalance_batch
from hlmicro.analytics.microprice import compute_microprice_batch
from hlmicro.analytics.spread import compute_spread_batch


def compute_order_flow_imbalance_batch(df: pl.DataFrame) -> pl.DataFrame:
    """Level-1 order-flow imbalance (OFI), Cont, Kukanov & Stoikov (2014),
    "The Price Impact of Order Book Events", J. Financial Econometrics.

    The flow analogue of static imbalance: net change in resting size at
    the best bid/ask between consecutive updates, where a price-level
    *change* (not just a size change) is treated as the old size vanishing
    / the new size appearing wholesale, not as a same-level size delta:

        dW_bid = q_bid                    if best_bid rose
                 q_bid - q_bid_prev        if best_bid unchanged
                 -q_bid_prev               if best_bid fell
        dW_ask = q_ask                    if best_ask fell
                 q_ask - q_ask_prev        if best_ask unchanged
                 -q_ask_prev               if best_ask rose
        OFI = dW_bid - dW_ask

    Requires df sorted by exch_time_ms and containing a SINGLE symbol -
    mixing symbols here would compute nonsense deltas across an arbitrary
    symbol boundary. The first row's OFI is always null (no prior state).
    """
    bb = df["best_bid"].to_numpy()
    bb_sz = df["bid_sz"].list.get(0).to_numpy()
    ba = df["best_ask"].to_numpy()
    ba_sz = df["ask_sz"].list.get(0).to_numpy()

    bb_prev = np.roll(bb, 1).astype(float)
    bb_sz_prev = np.roll(bb_sz, 1).astype(float)
    ba_prev = np.roll(ba, 1).astype(float)
    ba_sz_prev = np.roll(ba_sz, 1).astype(float)
    bb_prev[0] = np.nan
    bb_sz_prev[0] = np.nan
    ba_prev[0] = np.nan
    ba_sz_prev[0] = np.nan

    dW_bid = np.where(bb > bb_prev, bb_sz, np.where(bb == bb_prev, bb_sz - bb_sz_prev, -bb_sz_prev))
    dW_ask = np.where(ba < ba_prev, ba_sz, np.where(ba == ba_prev, ba_sz - ba_sz_prev, -ba_sz_prev))
    ofi = dW_bid - dW_ask
    ofi[0] = np.nan

    # fill_nan(None): a float NaN is a normal value to polars (survives
    # .drop_nulls()), so "no prior state" must be encoded as a real null.
    return df.with_columns(pl.Series("ofi", ofi).fill_nan(None))


def add_realized_vol_and_momentum(df: pl.DataFrame, lookback_updates: int = 20) -> pl.DataFrame:
    """Backward-looking (causal) realized vol and momentum controls, over
    the trailing `lookback_updates` book updates. Single-symbol input."""
    log_mid = np.log(df["mid"].to_numpy())
    log_ret = np.diff(log_mid, prepend=np.nan)
    df = df.with_columns(pl.Series("log_ret_1", log_ret).fill_nan(None))
    min_samples = max(2, lookback_updates // 4)
    realized_vol = df["log_ret_1"].rolling_std(
        window_size=lookback_updates, min_samples=min_samples
    )
    momentum = df["log_ret_1"].rolling_sum(window_size=lookback_updates, min_samples=min_samples)
    return df.with_columns(realized_vol.alias("realized_vol"), momentum.alias("momentum"))


def build_feature_table(
    l2book_df: pl.DataFrame,
    depth_levels: tuple[int, ...] = (5, 10, 20),
    vol_lookback_updates: int = 20,
) -> pl.DataFrame:
    """Full causal feature set for one symbol's sorted l2book history:
    spread, microprice/mid deviation, top-of-book + depth-weighted
    imbalance, order-flow imbalance, realized vol, momentum."""
    coins = l2book_df["coin"].unique().to_list()
    if len(coins) > 1:
        raise ValueError(f"build_feature_table expects a single symbol, got {coins}")

    df = l2book_df.sort("exch_time_ms").unique(subset=["exch_time_ms"], keep="first")
    df = compute_spread_batch(df)
    df = compute_microprice_batch(df)
    df = compute_imbalance_batch(df, depth_levels=depth_levels)
    df = compute_order_flow_imbalance_batch(df)
    df = add_realized_vol_and_momentum(df, lookback_updates=vol_lookback_updates)
    return df
