#!/usr/bin/env python
"""Run a backtest for one strategy against stored, normalized data.

    python scripts/run_backtest.py --strategy inventory_mm --symbol BTC
    python scripts/run_backtest.py --strategy naive_mm --symbol BTC --out reports/

Writes a JSON summary + an equity-curve PNG per run.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from hlmicro.analytics.imbalance import compute_imbalance_batch  # noqa: E402
from hlmicro.analytics.liquidity import (  # noqa: E402
    compute_depth_within_bps_batch,
    flag_liquidity_drought,
    rolling_liquidity_zscore,
)
from hlmicro.analytics.microprice import compute_microprice_batch  # noqa: E402
from hlmicro.analytics.spread import compute_spread_batch  # noqa: E402
from hlmicro.backtest.engine import BacktestConfig, BacktestEngine  # noqa: E402
from hlmicro.config import load_config  # noqa: E402
from hlmicro.strategies.inventory_mm import InventoryAwareMM  # noqa: E402
from hlmicro.strategies.naive_mm import NaiveSymmetricMM  # noqa: E402

logger = logging.getLogger("run_backtest")


def _load_table(processed_dir: Path, table: str, symbol: str) -> pl.DataFrame:
    files = sorted(glob.glob(str(processed_dir / table / "*" / "*.parquet")))
    if not files:
        return pl.DataFrame()
    df = pl.concat([pl.read_parquet(f) for f in files])
    return df.filter(pl.col("coin") == symbol)


def prepare_l2book(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    df = df.sort("exch_time_ms").unique(subset=["exch_time_ms"], keep="first")
    df = compute_spread_batch(df)
    df = compute_microprice_batch(df)
    df = compute_imbalance_batch(
        df, depth_levels=tuple(cfg["analytics"]["imbalance"]["depth_weighted_levels"])
    )
    df = compute_depth_within_bps_batch(
        df, bps_window=cfg["analytics"]["liquidity"]["depth_window_bps"]
    )
    depth_col = f"depth_within_{int(cfg['analytics']['liquidity']['depth_window_bps'])}bps"
    z = rolling_liquidity_zscore(
        df[depth_col], window=cfg["analytics"]["liquidity"]["rolling_window_s"]
    )
    drought = flag_liquidity_drought(
        z, threshold=cfg["analytics"]["liquidity"]["z_score_threshold"]
    )
    return df.with_columns(drought.alias("liquidity_drought"))


def build_strategy(name: str, cfg: dict):
    if name == "naive_mm":
        c = cfg["strategies"]["naive_mm"]
        return NaiveSymmetricMM(
            half_spread_bps=c["half_spread_bps"],
            quote_size=c["quote_size"],
            max_inventory=c["max_inventory"],
        )
    if name == "inventory_mm":
        c = cfg["strategies"]["inventory_mm"]
        return InventoryAwareMM(
            risk_aversion=c["risk_aversion"],
            vol_lookback_s=c["vol_lookback_s"],
            decay_window_s=c["decay_window_s"],
            max_inventory=c["max_inventory"],
            quote_size=c["quote_size"],
            base_half_spread_bps=c["base_half_spread_bps"],
        )
    raise ValueError(f"Unknown strategy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=["naive_mm", "inventory_mm"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    cfg = load_config(args.config)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    l2book = _load_table(processed_dir, "l2book", args.symbol)
    trades = _load_table(processed_dir, "trades", args.symbol).sort("exch_time_ms")
    asset_ctx = _load_table(processed_dir, "asset_ctx", args.symbol).sort("recv_ts_ns")
    if l2book.height == 0:
        raise SystemExit(f"No l2book data for {args.symbol} under {processed_dir}")
    logger.info(
        "Loaded %s: l2book=%d trades=%d asset_ctx=%d",
        args.symbol,
        l2book.height,
        trades.height,
        asset_ctx.height,
    )

    l2book = prepare_l2book(l2book, cfg)
    strategy = build_strategy(args.strategy, cfg)
    engine = BacktestEngine(
        strategy,
        symbol=args.symbol,
        config=BacktestConfig(
            maker_fee_bps=cfg["fees"]["maker_bps"],
            starting_cash=cfg["backtest"]["starting_cash_usd"],
            min_consumption_pct=cfg["backtest"]["fill_heuristic"]["min_trade_consumption_pct"],
        ),
    )
    report = engine.run(l2book, trades, asset_ctx)

    summary = {
        "strategy": report.strategy_name,
        "symbol": report.symbol,
        "n_fills": report.n_fills,
        "turnover_usd": report.turnover_usd,
        "realized_pnl": report.realized_pnl,
        "fees_paid": report.fees_paid,
        "funding_paid": report.funding_paid,
        "starting_cash": report.starting_cash,
        "final_equity": report.final_equity,
        "total_return_pct": (report.final_equity / report.starting_cash - 1) * 100,
        "sharpe": report.sharpe,
        "max_drawdown_pct": report.max_drawdown * 100,
        "var_95": report.var_95,
        "es_95": report.es_95,
    }
    logger.info("Result: %s", json.dumps(summary, indent=2))

    stem = f"{args.strategy}_{args.symbol}"
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    report.fills.write_csv(out_dir / f"{stem}_fills.csv")
    report.equity_curve.write_csv(out_dir / f"{stem}_equity.csv")

    if report.equity_curve.height > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ec = report.equity_curve
        ax.plot(ec["timestamp_ms"], ec["equity"])
        ax.set_title(f"{report.strategy_name} - {report.symbol} equity curve")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("equity (USD)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_equity.png", dpi=120)
        plt.close(fig)

    logger.info("Wrote report to %s", out_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
