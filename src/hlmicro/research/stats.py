"""Statistical methods for the imbalance -> forward-return study:
IC (Pearson/Spearman), HAC-SE predictive regression, directional accuracy
with a binomial test, a net-of-cost sanity check, and the chronological
train/test split that disciplines all of the above.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy.stats import binomtest, pearsonr, spearmanr, ttest_1samp


def chronological_split(
    df: pl.DataFrame, train_fraction: float, time_col: str = "exch_time_ms"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split by TIME (not row count) so an uneven update cadence doesn't
    distort the split point: everything up to the fraction-weighted
    timestamp is train, everything after is test. Never shuffled."""
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    cutoff = t_min + train_fraction * (t_max - t_min)
    train = df.filter(pl.col(time_col) <= cutoff)
    test = df.filter(pl.col(time_col) > cutoff)
    return train, test


def information_coefficient(feature: np.ndarray, label: np.ndarray) -> dict:
    mask = ~(np.isnan(feature) | np.isnan(label))
    f, l = feature[mask], label[mask]  # noqa: E741
    if len(f) < 10 or np.std(f) == 0 or np.std(l) == 0:
        return {
            "n": int(len(f)),
            "pearson": float("nan"),
            "pearson_p": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
        }
    pear_r, pear_p = pearsonr(f, l)
    spear_r, spear_p = spearmanr(f, l)
    return {
        "n": int(len(f)),
        "pearson": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman": float(spear_r),
        "spearman_p": float(spear_p),
    }


def newey_west_maxlags(horizon_ms: float, median_sampling_interval_ms: float) -> int:
    """Overlapping-window labels (a forward return over horizon h, sampled
    more often than every h) induce autocorrelation in the regression
    residuals out to roughly h / (sampling interval) lags - the number of
    consecutive observations that share part of the same forward window.
    HAC/Newey-West maxlags is set to that (floored at 1, capped at 500 to
    keep the correction itself numerically stable on large samples)."""
    if median_sampling_interval_ms <= 0:
        return 1
    return int(np.clip(np.ceil(horizon_ms / median_sampling_interval_ms), 1, 500))


def hac_regression(df: pl.DataFrame, y_col: str, x_cols: list[str], maxlags: int) -> dict:
    sub = df.select([y_col, *x_cols]).drop_nulls().to_pandas()
    if len(sub) < max(30, 5 * (len(x_cols) + 1)):
        return {
            "n": len(sub),
            "params": {},
            "tvalues": {},
            "pvalues": {},
            "r_squared": float("nan"),
        }
    y = sub[y_col]
    X = sm.add_constant(sub[x_cols])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "n": int(len(sub)),
        "params": model.params.to_dict(),
        "tvalues": model.tvalues.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "r_squared": float(model.rsquared),
    }


def directional_accuracy(feature: np.ndarray, label: np.ndarray) -> dict:
    mask = ~(np.isnan(feature) | np.isnan(label)) & (feature != 0) & (label != 0)
    f, l = feature[mask], label[mask]  # noqa: E741
    n = len(f)
    if n == 0:
        return {"n": 0, "hit_rate": float("nan"), "p_value": float("nan")}
    hits = int((np.sign(f) == np.sign(l)).sum())
    test = binomtest(hits, n, p=0.5)
    return {"n": n, "hit_rate": hits / n, "p_value": float(test.pvalue)}


def net_of_cost_check(
    df: pl.DataFrame,
    imbalance_col: str,
    label_col: str,
    spread_bps_col: str,
    threshold: float,
    taker_fee_bps: float,
) -> dict:
    """Toy backtest: enter in the direction of sign(imbalance) whenever
    |imbalance| > threshold, hold for the label's horizon, subtract a
    round-trip cost of (entry-time spread, i.e. half-spread paid on each
    leg = full spread) + two taker fees (this is a signal-driven, cross-
    the-spread entry/exit, not a passive quote - taker costs apply)."""
    sub = (
        df.filter(pl.col(imbalance_col).abs() > threshold)
        .select([imbalance_col, label_col, spread_bps_col])
        .drop_nulls()
    )
    if sub.height == 0:
        return {
            "n": 0,
            "mean_net_bps": float("nan"),
            "pct_positive": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    direction = np.sign(sub[imbalance_col].to_numpy())
    raw_ret_bps = sub[label_col].to_numpy() * 10_000 * direction
    cost_bps = sub[spread_bps_col].to_numpy() + 2 * taker_fee_bps
    net_bps = raw_ret_bps - cost_bps

    if len(net_bps) < 2 or np.std(net_bps) == 0:
        t_stat, p_val = float("nan"), float("nan")
    else:
        t_stat, p_val = ttest_1samp(net_bps, 0.0)

    return {
        "n": int(len(net_bps)),
        "mean_net_bps": float(np.mean(net_bps)),
        "pct_positive": float(np.mean(net_bps > 0)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }
