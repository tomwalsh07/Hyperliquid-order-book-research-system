"""Event-driven, strictly chronological backtest engine.

Replays stored l2book + trades + asset_ctx history in time order against a
Strategy, using FillSimulator for fills and PnLTracker for accounting.

Two documented simplifications beyond the fill heuristic itself
(see fills.py for that one):

1. **Quote prices are rounded to the symbol's tick size** before being
   handed to the fill simulator. A strategy computes a continuous price
   (e.g. microprice +/- a spread); real resting orders must sit at a valid
   tick. Bid prices round down, ask prices round up (never round *toward*
   the market), so rounding can only make us slightly more passive, never
   more aggressive than intended.
2. **Funding accrues continuously**, not at discrete hourly boundaries:
   each new activeAssetCtx observation applies
   `inventory * oracle_price * funding_rate * elapsed_hours_since_last_obs`.
   This converges to the same total as hourly settlement over an hour of
   roughly-constant rate, without needing to align to wall-clock hour
   boundaries against potentially gappy data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from hlmicro.analytics.funding import funding_payment
from hlmicro.backtest.costs import maker_fee
from hlmicro.backtest.fills import FillSimulator
from hlmicro.risk.pnl import (
    PnLTracker,
    expected_shortfall,
    max_drawdown,
    sharpe_ratio,
    value_at_risk,
)
from hlmicro.strategies.base import Fill, MarketState, Strategy

Side = Literal["bid", "ask"]


def infer_tick_size(prices: list[float]) -> float:
    """Smallest positive gap between sorted, de-duplicated prices in a
    single book snapshot's ladder - a robust-enough proxy for the symbol's
    price increment given a 20-level ladder."""
    uniq = sorted(set(prices))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 1e-12]
    return float(diffs.min()) if len(diffs) else 0.01


def _round_to_tick(price: float, tick: float, direction: Literal["down", "up"]) -> float:
    if tick <= 0:
        return price
    n = price / tick
    n = math.floor(n) if direction == "down" else math.ceil(n)
    return round(n * tick, 10)


@dataclass
class BacktestConfig:
    maker_fee_bps: float
    starting_cash: float = 100_000.0
    min_consumption_pct: float = 0.5
    periods_per_year_for_sharpe: float = 252 * 24 * 3600  # equity snapshots ~1/s scale; see report


@dataclass
class BacktestReport:
    strategy_name: str
    symbol: str
    n_fills: int
    turnover_usd: float
    realized_pnl: float
    fees_paid: float
    funding_paid: float
    final_equity: float
    starting_cash: float
    sharpe: float
    max_drawdown: float
    var_95: float
    es_95: float
    equity_curve: pl.DataFrame
    fills: pl.DataFrame


@dataclass
class _EngineState:
    tick_size: float | None = None
    open_size: dict[Side, float] = field(default_factory=lambda: {"bid": 0.0, "ask": 0.0})
    last_funding_ts_ms: int | None = None
    last_mid: float | None = None
    fills: list[dict] = field(default_factory=list)
    equity_rows: list[dict] = field(default_factory=list)


class BacktestEngine:
    def __init__(self, strategy: Strategy, symbol: str, config: BacktestConfig) -> None:
        self.strategy = strategy
        self.symbol = symbol
        self.config = config
        self.pnl = PnLTracker(starting_cash=config.starting_cash)
        self.fill_sim = FillSimulator(min_consumption_pct=config.min_consumption_pct)
        self.state = _EngineState()

    def run(
        self, l2book_df: pl.DataFrame, trades_df: pl.DataFrame, asset_ctx_df: pl.DataFrame
    ) -> BacktestReport:
        events = self._merge_events(l2book_df, trades_df, asset_ctx_df)
        for _ts, _prio, kind, row in events:
            if kind == "book":
                self._on_book(row)
            elif kind == "trade":
                self._on_trade(row)
            else:
                self._on_funding(row)

        return self._build_report()

    @staticmethod
    def _merge_events(
        l2book_df: pl.DataFrame, trades_df: pl.DataFrame, asset_ctx_df: pl.DataFrame
    ) -> list:
        events = []
        for row in l2book_df.iter_rows(named=True):
            events.append((row["exch_time_ms"], 0, "book", row))
        for row in trades_df.iter_rows(named=True):
            events.append((row["exch_time_ms"], 1, "trade", row))
        for row in asset_ctx_df.iter_rows(named=True):
            ts_ms = row["recv_ts_ns"] // 1_000_000
            events.append((ts_ms, 2, "funding", row))
        events.sort(key=lambda e: (e[0], e[1]))
        return events

    def _on_book(self, row: dict) -> None:
        if self.state.tick_size is None:
            self.state.tick_size = infer_tick_size(row["ask_px"])
        self.state.last_mid = row["mid"]

        state = MarketState(
            coin=row["coin"],
            timestamp_ms=row["exch_time_ms"],
            best_bid=row["best_bid"],
            best_bid_sz=row["bid_sz"][0],
            best_ask=row["best_ask"],
            best_ask_sz=row["ask_sz"][0],
            mid=row["mid"],
            microprice=row["microprice"],
            tob_imbalance=row["tob_imbalance"],
            liquidity_drought=bool(row.get("liquidity_drought", False)),
        )
        self.strategy.on_book_update(state)
        quote = self.strategy.get_quotes(state)

        for side, px, sz, ladder_px, ladder_sz, direction in (
            ("bid", quote.bid_px, quote.bid_sz, row["bid_px"], row["bid_sz"], "down"),
            ("ask", quote.ask_px, quote.ask_sz, row["ask_px"], row["ask_sz"], "up"),
        ):
            if px is None or sz <= 0:
                self.fill_sim.set_quote(side, None, None)
                self.state.open_size[side] = 0.0
                continue
            rounded = _round_to_tick(px, self.state.tick_size, direction)
            depth = 0.0
            for lp, ls in zip(ladder_px, ladder_sz, strict=True):
                if lp == rounded:
                    depth = ls
                    break
            self.fill_sim.set_quote(side, rounded, depth)
            self.state.open_size[side] = sz

        self.state.equity_rows.append(
            {
                "timestamp_ms": row["exch_time_ms"],
                "equity": self.pnl.equity(row["mid"]),
                "inventory": self.pnl.inventory,
                "cash": self.pnl.cash,
                "mid": row["mid"],
            }
        )

    def _on_trade(self, row: dict) -> None:
        fills = self.fill_sim.on_trade(row["side"], row["px"], row["sz"])
        for side, size in fills.items():
            capped = min(size, self.state.open_size[side])
            if capped <= 0:
                continue
            self.state.open_size[side] -= capped
            price = self.fill_sim.resting_price(side)
            our_side: Literal["buy", "sell"] = "buy" if side == "bid" else "sell"
            fee = maker_fee(price * capped, self.config.maker_fee_bps)

            fill = Fill(timestamp_ms=row["exch_time_ms"], side=our_side, price=price, size=capped)
            self.strategy.on_fill(fill)
            self.pnl.apply_fill(our_side, price, capped, fee)
            self.state.fills.append(
                {
                    "timestamp_ms": row["exch_time_ms"],
                    "side": our_side,
                    "price": price,
                    "size": capped,
                    "fee": fee,
                }
            )

    def _on_funding(self, row: dict) -> None:
        ts_ms = row["recv_ts_ns"] // 1_000_000
        if self.state.last_funding_ts_ms is None:
            self.state.last_funding_ts_ms = ts_ms
            return
        elapsed_hours = (ts_ms - self.state.last_funding_ts_ms) / 3_600_000
        self.state.last_funding_ts_ms = ts_ms
        if elapsed_hours <= 0 or self.pnl.inventory == 0:
            return
        amount = (
            funding_payment(self.pnl.inventory, row["oracle_px"], row["funding"]) * elapsed_hours
        )
        self.pnl.apply_funding(amount)

    def _build_report(self) -> BacktestReport:
        equity_df = pl.DataFrame(self.state.equity_rows)
        fills_df = (
            pl.DataFrame(self.state.fills)
            if self.state.fills
            else pl.DataFrame(
                schema={
                    "timestamp_ms": pl.Int64,
                    "side": pl.Utf8,
                    "price": pl.Float64,
                    "size": pl.Float64,
                    "fee": pl.Float64,
                }
            )
        )

        # final_equity always reflects the true final state (last known mark
        # price after ALL events, including trades after the last book
        # snapshot) - never derived from equity_rows, which only samples at
        # book-update ticks and would otherwise understate a fill that
        # landed after the final book snapshot in this replay window.
        final_equity = (
            self.pnl.equity(self.state.last_mid)
            if self.state.last_mid is not None
            else self.config.starting_cash
        )

        if equity_df.height > 1:
            equity = equity_df["equity"].to_numpy()
            returns = np.diff(equity)
            ts = equity_df["timestamp_ms"].to_numpy()
            elapsed_s = max((ts[-1] - ts[0]) / 1000, 1.0)
            periods_per_year = len(returns) / elapsed_s * 3600 * 24 * 365
            sharpe = sharpe_ratio(returns, periods_per_year)
            mdd = max_drawdown(equity)
            var95 = value_at_risk(returns, 0.95)
            es95 = expected_shortfall(returns, 0.95)
        else:
            sharpe = mdd = var95 = es95 = 0.0

        turnover = (
            float(fills_df["price"].to_numpy() @ fills_df["size"].to_numpy())
            if fills_df.height
            else 0.0
        )

        return BacktestReport(
            strategy_name=self.strategy.name,
            symbol=self.symbol,
            n_fills=fills_df.height,
            turnover_usd=turnover,
            realized_pnl=self.pnl.realized_pnl,
            fees_paid=self.pnl.fees_paid,
            funding_paid=self.pnl.funding_paid,
            final_equity=final_equity,
            starting_cash=self.config.starting_cash,
            sharpe=sharpe,
            max_drawdown=mdd,
            var_95=var95,
            es_95=es95,
            equity_curve=equity_df,
            fills=fills_df,
        )
