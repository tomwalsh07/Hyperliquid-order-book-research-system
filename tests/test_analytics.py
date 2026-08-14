import polars as pl
import pytest

from hlmicro.analytics.funding import (
    annualize_hourly_rate,
    compute_annualized_funding_batch,
    funding_payment,
)
from hlmicro.analytics.imbalance import (
    compute_imbalance_batch,
    depth_weighted_imbalance,
    level_weights,
    top_of_book_imbalance,
)
from hlmicro.analytics.liquidity import (
    compute_depth_within_bps_batch,
    depth_within_bps,
    flag_liquidity_drought,
    rolling_liquidity_zscore,
)
from hlmicro.analytics.microprice import compute_microprice_batch, microprice
from hlmicro.analytics.slippage import compute_slippage_batch, estimate_slippage, walk_the_book
from hlmicro.analytics.spread import compute_spread_batch, spread_abs, spread_bps

# ---- spread ----------------------------------------------------------


def test_spread_abs_and_bps_hand_computed():
    # bid=100, ask=101 -> abs=1, mid=100.5, bps = 1/100.5*10000 = 99.502487...
    assert spread_abs(100.0, 101.0) == 1.0
    assert spread_bps(100.0, 101.0) == pytest.approx(99.502487, rel=1e-6)


def test_spread_batch_matches_scalar():
    df = pl.DataFrame(
        {
            "bid_px": [[100.0, 99.0]],
            "ask_px": [[101.0, 102.0]],
            "bid_sz": [[1.0, 1.0]],
            "ask_sz": [[1.0, 1.0]],
        }
    )
    out = compute_spread_batch(df)
    row = out.row(0, named=True)
    assert row["best_bid"] == 100.0
    assert row["best_ask"] == 101.0
    assert row["mid"] == 100.5
    assert row["spread_bps"] == pytest.approx(spread_bps(100.0, 101.0))


# ---- microprice --------------------------------------------------------


def test_microprice_hand_computed():
    # micro = (100*1 + 101*2) / (2+1) = 302/3 = 100.66666...
    m = microprice(best_bid=100.0, best_bid_sz=2.0, best_ask=101.0, best_ask_sz=1.0)
    assert m == pytest.approx(302 / 3)


def test_microprice_falls_back_to_mid_when_no_size():
    assert microprice(100.0, 0.0, 102.0, 0.0) == 101.0


def test_microprice_batch_matches_scalar():
    df = pl.DataFrame(
        {"bid_px": [[100.0]], "ask_px": [[101.0]], "bid_sz": [[2.0]], "ask_sz": [[1.0]]}
    )
    df = compute_spread_batch(df)
    out = compute_microprice_batch(df)
    assert out.row(0, named=True)["microprice"] == pytest.approx(302 / 3)


# ---- imbalance -----------------------------------------------------------


def test_top_of_book_imbalance_hand_computed():
    # (2-1)/(2+1) = 1/3
    assert top_of_book_imbalance(2.0, 1.0) == pytest.approx(1 / 3)


def test_top_of_book_imbalance_empty_book_is_zero():
    assert top_of_book_imbalance(0.0, 0.0) == 0.0


def test_level_weights_linear_decay():
    assert level_weights(3, "linear_decay").tolist() == [3.0, 2.0, 1.0]


def test_depth_weighted_imbalance_hand_computed():
    # weights [3,2,1]; bid=[1,2,3] -> bid_depth=3*1+2*2+1*3=10
    # ask=[3,2,1] -> ask_depth=3*3+2*2+1*1=14; imbalance=(10-14)/24=-1/6
    val = depth_weighted_imbalance([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], n=3, scheme="linear_decay")
    assert val == pytest.approx(-1 / 6)


def test_depth_weighted_imbalance_pads_short_books():
    # only 2 levels available but n=3 requested -> padded with 0
    val = depth_weighted_imbalance([1.0, 2.0], [1.0, 2.0], n=3, scheme="uniform")
    # bid_depth = 1+2+0=3, ask_depth=1+2+0=3 -> imbalance 0
    assert val == pytest.approx(0.0)


def test_imbalance_batch_matches_scalar():
    df = pl.DataFrame(
        {
            "bid_sz": [[1.0, 2.0, 3.0]],
            "ask_sz": [[3.0, 2.0, 1.0]],
        }
    )
    out = compute_imbalance_batch(df, depth_levels=(3,))
    row = out.row(0, named=True)
    assert row["tob_imbalance"] == pytest.approx(top_of_book_imbalance(1.0, 3.0))
    assert row["imbalance_d3"] == pytest.approx(-1 / 6)


# ---- slippage --------------------------------------------------------------


def test_walk_the_book_partial_and_full_fill():
    levels = [(101.0, 1.0), (102.0, 2.0), (103.0, 5.0)]
    # order 3: take 1@101 + 2@102 -> notional=101+204=305, vwap=305/3
    vwap, filled = walk_the_book(levels, order_size=3.0)
    assert filled == 3.0
    assert vwap == pytest.approx(305 / 3)

    # order bigger than total visible depth (1+2+5=8) -> partial fill
    vwap2, filled2 = walk_the_book(levels, order_size=10.0)
    assert filled2 == 8.0
    total_notional = 101 * 1 + 102 * 2 + 103 * 5
    assert vwap2 == pytest.approx(total_notional / 8)


def test_estimate_slippage_buy_side_hand_computed():
    levels = [(101.0, 1.0), (102.0, 2.0)]
    res = estimate_slippage(levels, order_size=3.0, mid=100.5, side="buy")
    vwap = (101 * 1 + 102 * 2) / 3
    expected_bps = (vwap - 100.5) / 100.5 * 10_000
    assert res["vwap"] == pytest.approx(vwap)
    assert res["slippage_bps"] == pytest.approx(expected_bps)
    assert res["fully_filled"] is True


def test_compute_slippage_batch_matches_scalar():
    df = pl.DataFrame(
        {
            "ask_px": [[101.0, 102.0]],
            "ask_sz": [[1.0, 2.0]],
            "bid_px": [[99.0, 98.0]],
            "bid_sz": [[1.0, 2.0]],
            "mid": [100.5],
        }
    )
    out = compute_slippage_batch(df, order_sizes=[3.0], side="buy")
    expected = estimate_slippage([(101.0, 1.0), (102.0, 2.0)], 3.0, 100.5, "buy")
    assert out.row(0, named=True)["slippage_bps_3"] == pytest.approx(expected["slippage_bps"])


# ---- liquidity ----------------------------------------------------------


def test_depth_within_bps_hand_computed():
    # mid=100, window=100bps -> threshold = 1.0 price unit
    # bids within [99,100]: px=99.5(sz2), px=99(sz3) both within; px=98 excluded
    # asks within [100,101]: px=100.5(sz1), px=101(sz4) both within; px=102 excluded
    depth = depth_within_bps(
        bid_px=[99.5, 99.0, 98.0],
        bid_sz=[2.0, 3.0, 10.0],
        ask_px=[100.5, 101.0, 102.0],
        ask_sz=[1.0, 4.0, 10.0],
        mid=100.0,
        bps_window=100.0,
    )
    assert depth == pytest.approx(2 + 3 + 1 + 4)


def test_depth_within_bps_batch_matches_scalar():
    df = pl.DataFrame(
        {
            "bid_px": [[99.5, 99.0, 98.0]],
            "bid_sz": [[2.0, 3.0, 10.0]],
            "ask_px": [[100.5, 101.0, 102.0]],
            "ask_sz": [[1.0, 4.0, 10.0]],
            "mid": [100.0],
        }
    )
    out = compute_depth_within_bps_batch(df, bps_window=100.0)
    assert out.row(0, named=True)["depth_within_100bps"] == pytest.approx(10.0)


def test_rolling_liquidity_zscore_and_drought_flag():
    # stable baseline (15 pts at 100) then a sharp, sustained collapse to ~5
    depth = pl.Series("depth", [100.0] * 15 + [98.0, 5.0, 4.0, 6.0, 5.0])
    z = rolling_liquidity_zscore(depth, window=10)
    drought = flag_liquidity_drought(z, threshold=-2.5)
    # collapse starts at index 15; that point and the next should trip the flag
    assert drought[15] and drought[16]
    # the flat, stable baseline itself must never be flagged
    assert not any(drought[:15])


# ---- funding --------------------------------------------------------------


def test_annualize_hourly_rate_matches_hyperliquid_doc_example():
    # docs: 0.00125%/hour is quoted as "11.6% APR" (compounding, not simple mult)
    apr = annualize_hourly_rate(0.0000125)
    assert apr == pytest.approx(0.1157, abs=0.001)


def test_funding_payment_formula():
    # payment = position_size * oracle_price * funding_rate
    assert funding_payment(
        position_size=2.0, oracle_price=100.0, funding_rate=0.0001
    ) == pytest.approx(0.02)


def test_compute_annualized_funding_batch():
    df = pl.DataFrame({"funding": [0.0000125]})
    out = compute_annualized_funding_batch(df)
    assert out.row(0, named=True)["funding_apr"] == pytest.approx(annualize_hourly_rate(0.0000125))
