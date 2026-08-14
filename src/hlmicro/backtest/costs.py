"""Trading costs: fees (maker/taker, config-driven, never hardcoded) and
funding accrual (reuses analytics.funding's docs-confirmed formula)."""

from __future__ import annotations

from hlmicro.analytics.funding import funding_payment

__all__ = ["maker_fee", "taker_fee", "funding_payment"]


def maker_fee(notional: float, maker_bps: float) -> float:
    return abs(notional) * maker_bps / 10_000


def taker_fee(notional: float, taker_bps: float) -> float:
    return abs(notional) * taker_bps / 10_000
