"""Funding rate tracking: current (from activeAssetCtx) + predicted (from
the polled predictedFundings endpoint), annualized.

Annualization uses compounding, `(1 + hourly_rate) ** 8760 - 1`, not simple
multiplication (`hourly_rate * 8760`). This matches Hyperliquid's own
documented example: 0.00125%/hour is quoted as "11.6% APR" - simple
multiplication gives 10.95%, compounding gives 11.57% (~11.6%), confirming
which convention they use. See docs/api_notes.md §4.
"""

from __future__ import annotations

import polars as pl

HOURS_PER_YEAR = 24 * 365


def annualize_hourly_rate(hourly_rate: float) -> float:
    return (1 + hourly_rate) ** HOURS_PER_YEAR - 1


def funding_payment(position_size: float, oracle_price: float, funding_rate: float) -> float:
    """funding_payment = position_size * oracle_price * funding_rate
    (docs-confirmed formula - uses oracle price, not mark price). Positive
    position_size = long; a positive result is what the long side PAYS."""
    return position_size * oracle_price * funding_rate


def compute_annualized_funding_batch(df: pl.DataFrame, rate_col: str = "funding") -> pl.DataFrame:
    return df.with_columns(((1 + pl.col(rate_col)) ** HOURS_PER_YEAR - 1).alias(f"{rate_col}_apr"))


class FundingTracker:
    """Combines the two funding series that matter: current
    (last-settled, pushed continuously via activeAssetCtx) and predicted
    (Hyperliquid's own "HlPerp" venue in the polled predictedFundings
    response). Kept deliberately simple - a thin latest-value cache per
    symbol, not a time-series store (that's what the Parquet tables are for;
    this is for a live strategy/dashboard that wants "what's funding right
    now" without re-querying storage)."""

    def __init__(self) -> None:
        self._current: dict[str, float] = {}
        self._predicted: dict[str, float] = {}

    def update_current(self, coin: str, funding_rate: float) -> None:
        self._current[coin] = funding_rate

    def update_predicted(self, coin: str, venue: str, funding_rate: float) -> None:
        if venue == "HlPerp":
            self._predicted[coin] = funding_rate

    def current(self, coin: str) -> float | None:
        return self._current.get(coin)

    def predicted(self, coin: str) -> float | None:
        return self._predicted.get(coin)

    def current_apr(self, coin: str) -> float | None:
        r = self.current(coin)
        return annualize_hourly_rate(r) if r is not None else None

    def predicted_apr(self, coin: str) -> float | None:
        r = self.predicted(coin)
        return annualize_hourly_rate(r) if r is not None else None
