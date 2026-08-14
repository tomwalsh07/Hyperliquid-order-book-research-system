import math

import numpy as np
import pytest

from hlmicro.risk.pnl import (
    PnLTracker,
    expected_shortfall,
    max_drawdown,
    sharpe_ratio,
    value_at_risk,
)

# ---- PnLTracker (average-cost accounting) --------------------------------


def test_open_and_close_long_realizes_correct_pnl():
    pnl = PnLTracker(starting_cash=1000.0)
    pnl.apply_fill("buy", price=100.0, size=1.0, fee=0.1)
    assert pnl.inventory == 1.0
    assert pnl.avg_entry_price == 100.0
    assert pnl.cash == pytest.approx(1000.0 - 100.0 - 0.1)

    pnl.apply_fill("sell", price=105.0, size=1.0, fee=0.1)
    assert pnl.inventory == 0.0
    # realized = (105-100)*1 = 5
    assert pnl.realized_pnl == pytest.approx(5.0)
    assert pnl.fees_paid == pytest.approx(0.2)
    # cash: -100-0.1 (buy) +105-0.1 (sell) = 1000 + 4.8
    assert pnl.cash == pytest.approx(1000.0 + 4.8)
    assert pnl.equity(mark_price=105.0) == pytest.approx(pnl.cash)  # flat -> equity = cash


def test_averaging_into_a_larger_long_position():
    pnl = PnLTracker(starting_cash=0.0)
    pnl.apply_fill("buy", price=100.0, size=1.0, fee=0.0)
    pnl.apply_fill("buy", price=110.0, size=1.0, fee=0.0)
    # avg entry = (100*1 + 110*1)/2 = 105
    assert pnl.inventory == 2.0
    assert pnl.avg_entry_price == pytest.approx(105.0)


def test_partial_close_realizes_proportional_pnl_and_keeps_remainder():
    pnl = PnLTracker(starting_cash=0.0)
    pnl.apply_fill("buy", price=100.0, size=4.0, fee=0.0)
    pnl.apply_fill("sell", price=110.0, size=1.0, fee=0.0)
    # realized on the 1 unit closed: (110-100)*1 = 10; 3 units remain @ entry 100
    assert pnl.realized_pnl == pytest.approx(10.0)
    assert pnl.inventory == 3.0
    assert pnl.avg_entry_price == pytest.approx(100.0)


def test_flip_through_zero_resets_avg_entry_to_flip_price():
    pnl = PnLTracker(starting_cash=0.0)
    pnl.apply_fill("buy", price=100.0, size=2.0, fee=0.0)  # long 2 @ 100
    pnl.apply_fill("sell", price=110.0, size=5.0, fee=0.0)  # closes 2, opens short 3 @ 110
    # realized on closing the 2 longs: (110-100)*2 = 20
    assert pnl.realized_pnl == pytest.approx(20.0)
    assert pnl.inventory == -3.0
    assert pnl.avg_entry_price == pytest.approx(110.0)


def test_short_position_unrealized_pnl_sign():
    pnl = PnLTracker(starting_cash=0.0)
    pnl.apply_fill("sell", price=100.0, size=1.0, fee=0.0)  # short 1 @ 100
    # price drops to 90 -> short position profits by 10
    assert pnl.unrealized_pnl(mark_price=90.0) == pytest.approx(10.0)
    # price rises to 110 -> short position loses 10
    assert pnl.unrealized_pnl(mark_price=110.0) == pytest.approx(-10.0)


def test_funding_reduces_cash_and_is_tracked_separately():
    pnl = PnLTracker(starting_cash=100.0)
    pnl.apply_funding(1.5)  # we paid 1.5 in funding
    assert pnl.cash == pytest.approx(98.5)
    assert pnl.funding_paid == pytest.approx(1.5)


# ---- risk stats ------------------------------------------------------------


def test_sharpe_ratio_hand_computed():
    # returns [0.04,-0.02,0.04,-0.02]: mean=0.01, pop-std=0.03
    returns = np.array([0.04, -0.02, 0.04, -0.02])
    expected = 0.01 / 0.03 * math.sqrt(252)
    assert sharpe_ratio(returns, periods_per_year=252) == pytest.approx(expected, rel=1e-6)


def test_sharpe_ratio_zero_variance_returns_zero_not_inf():
    returns = np.array([0.02, 0.02, 0.02])
    assert sharpe_ratio(returns, periods_per_year=252) == 0.0


def test_max_drawdown_hand_computed():
    # equity [100,110,105,90,95,120]; running max [100,110,110,110,110,120]
    # drawdowns: 0,0,-5/110,-20/110,-15/110,0 -> min = -20/110
    equity = np.array([100.0, 110.0, 105.0, 90.0, 95.0, 120.0])
    assert max_drawdown(equity) == pytest.approx(-20 / 110)


def test_max_drawdown_monotonic_increase_is_zero():
    equity = np.array([100.0, 105.0, 110.0, 120.0])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_value_at_risk_hand_computed_small_sample():
    # sorted [-0.03,-0.01,0.02,0.04]; median (confidence=0.5) via linear
    # interpolation index = 0.5*(4-1)=1.5 -> between -0.01 and 0.02
    # = -0.01 + 0.5*0.03 = 0.005 -> VaR (positive-loss convention) = -0.005
    returns = np.array([0.04, -0.01, -0.03, 0.02])
    assert value_at_risk(returns, confidence=0.5) == pytest.approx(-0.005)


def test_expected_shortfall_is_at_least_as_severe_as_var():
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0, scale=0.02, size=500)
    var = value_at_risk(returns, confidence=0.95)
    es = expected_shortfall(returns, confidence=0.95)
    assert es >= var - 1e-9


def test_risk_stats_handle_empty_input():
    empty = np.array([])
    assert sharpe_ratio(empty, 252) == 0.0
    assert max_drawdown(empty) == 0.0
    assert value_at_risk(empty) == 0.0
    assert expected_shortfall(empty) == 0.0
