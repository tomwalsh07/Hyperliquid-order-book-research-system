import polars as pl
import pytest

from hlmicro.backtest.engine import BacktestConfig, BacktestEngine, infer_tick_size
from hlmicro.strategies.naive_mm import NaiveSymmetricMM


def _l2book_row(ts_ms: int) -> dict:
    return {
        "coin": "BTC",
        "exch_time_ms": ts_ms,
        "bid_px": [100.0, 99.0, 98.0],
        "bid_sz": [10.0, 10.0, 10.0],
        "ask_px": [101.0, 102.0, 103.0],
        "ask_sz": [10.0, 10.0, 10.0],
        "best_bid": 100.0,
        "best_ask": 101.0,
        "mid": 100.5,
        "microprice": 100.5,  # equal sizes both sides -> microprice == mid
        "tob_imbalance": 0.0,
        "liquidity_drought": False,
    }


def test_infer_tick_size():
    # sorted [101,102,103.5,105] -> diffs [1, 1.5, 1.5] -> min positive diff = 1.0
    assert infer_tick_size([101.0, 102.0, 103.5, 105.0]) == 1.0


def test_engine_credits_fill_and_updates_pnl_end_to_end():
    l2book_df = pl.DataFrame([_l2book_row(1_000)])
    trades_df = pl.DataFrame(
        [
            # half_spread_bps=0 -> bid rounds down to 100.0, ask rounds up to 101.0
            # threshold at bid = 0.5 * resting_depth(10) = 5; trade of 6 crosses it,
            # fill = 6 - 5 = 1, capped at quote_size=1.0 -> fill == 1.0
            {"coin": "BTC", "side": "A", "px": 100.0, "sz": 6.0, "exch_time_ms": 1_500}
        ]
    )
    asset_ctx_df = pl.DataFrame(
        schema={"recv_ts_ns": pl.Int64, "oracle_px": pl.Float64, "funding": pl.Float64}
    )

    strategy = NaiveSymmetricMM(half_spread_bps=0.0, quote_size=1.0, max_inventory=10.0)
    engine = BacktestEngine(
        strategy, symbol="BTC", config=BacktestConfig(maker_fee_bps=1.5, starting_cash=10_000.0)
    )
    report = engine.run(l2book_df, trades_df, asset_ctx_df)

    assert report.n_fills == 1
    fill_row = report.fills.row(0, named=True)
    assert fill_row["side"] == "buy"  # trade side "A" hits our bid -> we bought
    assert fill_row["price"] == 100.0
    assert fill_row["size"] == 1.0

    assert strategy.inventory == 1.0
    # fee = 100 * 1.0 * 1.5bps = 0.015
    assert report.fees_paid == pytest.approx(0.015)
    # cash = 10000 - 100*1 - 0.015 = 9899.985; mark = last mid = 100.5
    # equity = cash + inventory*mark = 9899.985 + 100.5 = 10000.485
    # (bought at 100.0 but mark-to-market at mid 100.5 -> +0.5 unrealized)
    assert report.final_equity == pytest.approx(10_000.485)


def test_engine_no_fill_when_trade_below_threshold():
    l2book_df = pl.DataFrame([_l2book_row(1_000)])
    trades_df = pl.DataFrame(
        [
            {"coin": "BTC", "side": "A", "px": 100.0, "sz": 2.0, "exch_time_ms": 1_500}
        ]  # below 5.0 threshold
    )
    asset_ctx_df = pl.DataFrame(
        schema={"recv_ts_ns": pl.Int64, "oracle_px": pl.Float64, "funding": pl.Float64}
    )
    strategy = NaiveSymmetricMM(half_spread_bps=0.0, quote_size=1.0, max_inventory=10.0)
    engine = BacktestEngine(strategy, symbol="BTC", config=BacktestConfig(maker_fee_bps=1.5))
    report = engine.run(l2book_df, trades_df, asset_ctx_df)
    assert report.n_fills == 0
    assert strategy.inventory == 0.0


def test_engine_respects_max_inventory_by_withdrawing_quote():
    l2book_df = pl.DataFrame([_l2book_row(1_000), _l2book_row(2_000)])
    trades_df = pl.DataFrame(
        [
            {"coin": "BTC", "side": "A", "px": 100.0, "sz": 6.0, "exch_time_ms": 1_500},
            {"coin": "BTC", "side": "A", "px": 100.0, "sz": 6.0, "exch_time_ms": 2_500},
        ]
    )
    asset_ctx_df = pl.DataFrame(
        schema={"recv_ts_ns": pl.Int64, "oracle_px": pl.Float64, "funding": pl.Float64}
    )
    # max_inventory smaller than one fill -> after first fill (size 1), bid should be withdrawn
    strategy = NaiveSymmetricMM(half_spread_bps=0.0, quote_size=1.0, max_inventory=0.5)
    engine = BacktestEngine(strategy, symbol="BTC", config=BacktestConfig(maker_fee_bps=1.5))
    report = engine.run(l2book_df, trades_df, asset_ctx_df)
    assert report.n_fills == 1  # second trade's bid quote was withdrawn, no second fill
    assert strategy.inventory == 1.0
