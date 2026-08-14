"""Forward-return label construction - the ONLY place in this codebase
allowed to look at future data.

For each row at time t and horizon h, the label uses the price at the
first available observation with timestamp >= t + h. Since h > 0, that
observation's timestamp is always strictly greater than t: searchsorted
against a non-decreasing time array with target = t + h > t guarantees the
match index has time >= t + h > t. There is no path in this function that
can return a value from a row at or before t - see tests/test_labels.py
for a direct assertion of this against the row indices actually used, not
just a sign-anity check on the output.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def add_forward_return_labels(
    df: pl.DataFrame,
    horizons_ms: list[int],
    price_cols: tuple[str, ...] = ("mid", "microprice"),
) -> pl.DataFrame:
    """Single-symbol input, sorted by exch_time_ms. Adds
    fwd_logret_{price_col}_{h}ms for every (price_col, horizon) pair.
    Rows too close to the end of the series to have a valid forward
    observation get null labels (never a fabricated/padded value)."""
    coins = df["coin"].unique().to_list()
    if len(coins) > 1:
        raise ValueError(f"add_forward_return_labels expects a single symbol, got {coins}")

    times = df["exch_time_ms"].to_numpy()
    n = len(times)
    out = df

    for h in horizons_ms:
        target = times + h
        idx = np.searchsorted(times, target, side="left")
        valid = idx < n

        for col in price_cols:
            vals = df[col].to_numpy()
            fwd = np.full(n, np.nan)
            fwd[valid] = vals[idx[valid]]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret = np.log(fwd) - np.log(vals)
            # NaN (missing horizon) must become a real polars null, not a
            # float NaN - NaN is a normal float value to polars/pandas
            # and silently survives .drop_nulls(), which would poison
            # every downstream correlation/regression/stat with rows that
            # look like real (NaN) numbers instead of being excluded.
            out = out.with_columns(pl.Series(f"fwd_logret_{col}_{h}ms", log_ret).fill_nan(None))

    return out


def forward_index_map(times: np.ndarray, horizon_ms: int) -> np.ndarray:
    """Exposed separately (not inlined) so tests can assert the no-lookahead
    property directly against the index mapping itself, independent of
    whatever price values happen to be at those indices."""
    target = times + horizon_ms
    return np.searchsorted(times, target, side="left")
