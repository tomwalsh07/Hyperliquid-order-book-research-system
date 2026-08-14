#!/usr/bin/env python
"""Run the imbalance -> forward-return study end to end.

    python scripts/run_research.py --symbols BTC,ETH,SOL

For each symbol: builds the causal feature table, attaches forward-return
labels at all configured horizons (no-lookahead enforced in labels.py,
tested in tests/test_research_labels.py), splits chronologically (never
randomly), calibrates the net-of-cost threshold on the TRAIN split only,
and reports IC / HAC regression / directional accuracy / net-of-cost
results on the held-out TEST split only.

Outputs:
  reports/research_results.csv   - full results table
  docs/assets/imbalance_decay.png - correlation-vs-horizon decay chart
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from hlmicro.config import load_config  # noqa: E402
from hlmicro.research.features import build_feature_table  # noqa: E402
from hlmicro.research.labels import add_forward_return_labels  # noqa: E402
from hlmicro.research.stats import (  # noqa: E402
    chronological_split,
    directional_accuracy,
    hac_regression,
    information_coefficient,
    net_of_cost_check,
    newey_west_maxlags,
)

logger = logging.getLogger("run_research")

FEATURE_COLS = ["tob_imbalance", "imbalance_d5", "imbalance_d10", "imbalance_d20", "ofi"]
CONTROL_COLS = ["spread_bps", "realized_vol", "momentum"]
NET_COST_THRESHOLD_QUANTILE = 0.75


def _load_l2book(processed_dir: Path, symbol: str) -> pl.DataFrame:
    files = sorted(glob.glob(str(processed_dir / "l2book" / "*" / "*.parquet")))
    if not files:
        return pl.DataFrame()
    df = pl.concat([pl.read_parquet(f) for f in files])
    return df.filter(pl.col("coin") == symbol)


def analyze_symbol(symbol: str, l2book: pl.DataFrame, cfg: dict) -> tuple[list[dict], float]:
    horizons_ms = cfg["research"]["horizons_ms"]
    train_fraction = cfg["research"]["train_fraction"]
    taker_fee_bps = cfg["fees"]["taker_bps"]
    depth_levels = tuple(cfg["analytics"]["imbalance"]["depth_weighted_levels"])

    feat = build_feature_table(l2book, depth_levels=depth_levels)
    feat = add_forward_return_labels(
        feat, horizons_ms=horizons_ms, price_cols=("mid", "microprice")
    )

    n_raw = feat.height
    train, test = chronological_split(feat, train_fraction=train_fraction)
    logger.info(
        "%s: %d causal l2book-derived feature rows (n=%d train / n=%d test, chronological split)",
        symbol,
        n_raw,
        train.height,
        test.height,
    )

    times = feat["exch_time_ms"].to_numpy()
    median_interval_ms = float(np.median(np.diff(times))) if len(times) > 1 else 1.0

    # Net-of-cost threshold calibrated on TRAIN only, per feature.
    thresholds = {}
    for feature in FEATURE_COLS:
        vals = train[feature].drop_nulls().to_numpy()
        thresholds[feature] = (
            float(np.quantile(np.abs(vals), NET_COST_THRESHOLD_QUANTILE))
            if len(vals) > 0
            else float("nan")
        )

    rows = []
    for horizon_ms in horizons_ms:
        label_col = f"fwd_logret_mid_{horizon_ms}ms"
        maxlags = newey_west_maxlags(horizon_ms, median_interval_ms)

        for feature in FEATURE_COLS:
            feat_arr = test[feature].to_numpy()
            label_arr = test[label_col].to_numpy()

            ic = information_coefficient(feat_arr, label_arr)
            da = directional_accuracy(feat_arr, label_arr)
            base_reg = hac_regression(test, y_col=label_col, x_cols=[feature], maxlags=maxlags)
            full_reg = hac_regression(
                test, y_col=label_col, x_cols=[feature, *CONTROL_COLS], maxlags=maxlags
            )
            net = net_of_cost_check(
                test,
                imbalance_col=feature,
                label_col=label_col,
                spread_bps_col="spread_bps",
                threshold=thresholds[feature],
                taker_fee_bps=taker_fee_bps,
            )

            rows.append(
                {
                    "symbol": symbol,
                    "feature": feature,
                    "horizon_ms": horizon_ms,
                    "n": ic["n"],
                    "pearson": ic["pearson"],
                    "pearson_p": ic["pearson_p"],
                    "spearman": ic["spearman"],
                    "spearman_p": ic["spearman_p"],
                    "hit_rate": da["hit_rate"],
                    "hit_rate_p": da["p_value"],
                    "ols_beta": base_reg["params"].get(feature, float("nan")),
                    "ols_p": base_reg["pvalues"].get(feature, float("nan")),
                    "ols_r2": base_reg["r_squared"],
                    "ols_beta_ctrl": full_reg["params"].get(feature, float("nan")),
                    "ols_p_ctrl": full_reg["pvalues"].get(feature, float("nan")),
                    "ols_r2_ctrl": full_reg["r_squared"],
                    "net_cost_threshold": thresholds[feature],
                    "net_mean_bps": net["mean_net_bps"],
                    "net_pct_positive": net["pct_positive"],
                    "net_n": net["n"],
                    "net_p": net["p_value"],
                }
            )
    return rows, median_interval_ms


def plot_decay_chart(
    results: pl.DataFrame,
    out_path: Path,
    feature: str = "tob_imbalance",
    native_cadence_ms: float | None = None,
) -> None:
    sub = results.filter(pl.col("feature") == feature).sort(["symbol", "horizon_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for symbol in sub["symbol"].unique().sort().to_list():
        s = sub.filter(pl.col("symbol") == symbol)
        axes[0].plot(s["horizon_ms"], s["pearson"], marker="o", label=symbol)
        axes[1].plot(s["horizon_ms"], s["spearman"], marker="o", label=symbol)

    for ax, title in zip(axes, ["Pearson IC", "Spearman IC"], strict=True):
        ax.axhline(0, color="grey", linewidth=0.8)
        if native_cadence_ms:
            ax.axvline(native_cadence_ms, color="red", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("horizon (ms)")
        ax.set_ylabel("correlation with fwd mid log-return")
        ax.set_title(title)
        ax.legend()

    subtitle = f"{feature}: correlation vs. horizon (held-out test split)"
    if native_cadence_ms:
        subtitle += (
            f"\nred dashed line = median l2Book push interval "
            f"(~{native_cadence_ms / 1000:.1f}s) - horizons to its left mostly share the "
            "same underlying observation, see docs/methodology.md"
        )
    fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Wrote decay chart to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTC,ETH,SOL")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--assets-dir", default="docs/assets")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    cfg = load_config(args.config)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    all_rows = []
    cadences = []
    for symbol in symbols:
        l2book = _load_l2book(processed_dir, symbol)
        if l2book.height < 50:
            logger.warning("%s: only %d l2book rows, skipping (need >= 50)", symbol, l2book.height)
            continue
        rows, median_interval_ms = analyze_symbol(symbol, l2book, cfg)
        all_rows.extend(rows)
        cadences.append(median_interval_ms)

    if not all_rows:
        raise SystemExit("No symbols had enough data to analyze.")

    results = pl.DataFrame(all_rows)
    results_path = out_dir / "research_results.csv"
    results.write_csv(results_path)
    logger.info("Wrote %d result rows to %s", results.height, results_path)

    overall_cadence_ms = float(np.median(cadences)) if cadences else None
    plot_decay_chart(
        results,
        Path(args.assets_dir) / "imbalance_decay.png",
        feature="tob_imbalance",
        native_cadence_ms=overall_cadence_ms,
    )

    with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=50):
        print(
            results.select(
                [
                    "symbol",
                    "feature",
                    "horizon_ms",
                    "n",
                    "pearson",
                    "spearman",
                    "hit_rate",
                    "net_mean_bps",
                ]
            )
        )


if __name__ == "__main__":
    main()
