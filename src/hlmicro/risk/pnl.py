"""Inventory/PnL accounting (average-cost method) and basic risk stats.

`PnLTracker` is the single source of truth the backtest engine mutates on
every fill/funding accrual. Equity is always computed as `cash +
inventory * mark_price` (never accumulated incrementally), so it can never
drift out of sync with cash flows - `realized_pnl`/`fees_paid`/
`funding_paid` are reporting-only decomposition fields, not used to derive
equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from hlmicro.strategies.base import FillSide


@dataclass
class PnLTracker:
    starting_cash: float
    cash: float = field(init=False)
    inventory: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0

    def __post_init__(self) -> None:
        self.cash = self.starting_cash

    def apply_fill(self, side: FillSide, price: float, size: float, fee: float) -> None:
        signed = size if side == "buy" else -size
        self.cash -= fee
        self.fees_paid += fee
        self.cash -= signed * price

        old_inventory = self.inventory
        same_direction = old_inventory == 0 or (old_inventory > 0) == (signed > 0)
        if same_direction:
            new_qty = old_inventory + signed
            self.avg_entry_price = (
                price
                if old_inventory == 0
                else (self.avg_entry_price * abs(old_inventory) + price * abs(signed))
                / abs(new_qty)
            )
            self.inventory = new_qty
        else:
            closing = min(abs(signed), abs(old_inventory))
            direction = 1 if old_inventory > 0 else -1
            self.realized_pnl += direction * (price - self.avg_entry_price) * closing
            new_qty = old_inventory + signed
            self.inventory = new_qty
            if new_qty != 0 and (new_qty > 0) != (old_inventory > 0):
                self.avg_entry_price = price  # flipped through zero -> fresh position

    def apply_funding(self, amount: float) -> None:
        """amount > 0 = we paid funding, amount < 0 = we received it."""
        self.cash -= amount
        self.funding_paid += amount

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.inventory == 0:
            return 0.0
        direction = 1 if self.inventory > 0 else -1
        return direction * (mark_price - self.avg_entry_price) * abs(self.inventory)

    def equity(self, mark_price: float) -> float:
        return self.cash + self.inventory * mark_price


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Returns a negative fraction, e.g. -0.12 for a 12% peak-to-trough drop."""
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    running_max = np.where(running_max == 0, np.finfo(float).eps, running_max)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR, reported as a positive number (a loss magnitude)."""
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Mean loss beyond VaR (also positive-is-a-loss)."""
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


def rolling_inventory_volatility(inventory: pl.Series, window: int) -> pl.Series:
    return inventory.rolling_std(window_size=window, min_samples=max(2, window // 4))
