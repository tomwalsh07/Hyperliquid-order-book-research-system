import numpy as np
import polars as pl
import pytest

from hlmicro.research.features import (
    add_realized_vol_and_momentum,
    build_feature_table,
    compute_order_flow_imbalance_batch,
)


def test_ofi_first_row_is_null():
    df = pl.DataFrame(
        {
            "best_bid": [100.0, 100.0],
            "best_ask": [101.0, 101.0],
            "bid_sz": [[5.0], [5.0]],
            "ask_sz": [[5.0], [5.0]],
        }
    )
    out = compute_order_flow_imbalance_batch(df)
    # must be a real polars null (drop_nulls()-visible), not a float NaN -
    # see test_research_labels.test_tail_labels_are_real_polars_nulls_not_float_nan
    assert out["ofi"].null_count() == 1
    assert out["ofi"][0] is None


def test_ofi_hand_computed_price_flat_size_change():
    # bid price unchanged, size grows 5->8 -> dW_bid = 8-5 = 3
    # ask price unchanged, size shrinks 5->2 -> dW_ask = 2-5 = -3
    # OFI = dW_bid - dW_ask = 3 - (-3) = 6
    df = pl.DataFrame(
        {
            "best_bid": [100.0, 100.0],
            "best_ask": [101.0, 101.0],
            "bid_sz": [[5.0], [8.0]],
            "ask_sz": [[5.0], [2.0]],
        }
    )
    out = compute_order_flow_imbalance_batch(df)
    assert out["ofi"][1] == pytest.approx(6.0)


def test_ofi_hand_computed_price_improves():
    # best bid rises 100->101 -> dW_bid = new size in full = 10 (old level vanished from top)
    # best ask unchanged 105, size 5->5 -> dW_ask = 5-5 = 0
    # OFI = 10 - 0 = 10
    df = pl.DataFrame(
        {
            "best_bid": [100.0, 101.0],
            "best_ask": [105.0, 105.0],
            "bid_sz": [[3.0], [10.0]],
            "ask_sz": [[5.0], [5.0]],
        }
    )
    out = compute_order_flow_imbalance_batch(df)
    assert out["ofi"][1] == pytest.approx(10.0)


def test_ofi_hand_computed_price_worsens():
    # best bid falls 100->99 -> dW_bid = -old_size = -3 (old level fully consumed/dropped)
    # best ask rises 105->106 -> dW_ask = -old_ask_size = -5 (ask rising -> -q_ask_prev)
    # OFI = -3 - (-5) = 2
    df = pl.DataFrame(
        {
            "best_bid": [100.0, 99.0],
            "best_ask": [105.0, 106.0],
            "bid_sz": [[3.0], [7.0]],
            "ask_sz": [[5.0], [9.0]],
        }
    )
    out = compute_order_flow_imbalance_batch(df)
    assert out["ofi"][1] == pytest.approx(2.0)


def test_realized_vol_and_momentum_causal_and_hand_computed():
    # mid: 100,101,102,101 -> log returns: nan, ln(1.01), ln(102/101), ln(101/102)
    mids = [100.0, 101.0, 102.0, 101.0]
    df = pl.DataFrame({"mid": mids})
    out = add_realized_vol_and_momentum(df, lookback_updates=3)
    log_rets = out["log_ret_1"].to_list()
    assert log_rets[0] is None or np.isnan(log_rets[0])
    assert log_rets[1] == pytest.approx(np.log(101 / 100))
    assert log_rets[2] == pytest.approx(np.log(102 / 101))
    assert log_rets[3] == pytest.approx(np.log(101 / 102))
    # momentum at row3 = rolling sum of log returns over window 3 (rows 1..3)
    expected_momentum = log_rets[1] + log_rets[2] + log_rets[3]
    assert out["momentum"][3] == pytest.approx(expected_momentum)


def test_build_feature_table_rejects_multi_symbol_input():
    df = pl.DataFrame(
        {
            "coin": ["BTC", "ETH"],
            "exch_time_ms": [0, 100],
            "bid_px": [[100.0], [200.0]],
            "bid_sz": [[1.0], [1.0]],
            "ask_px": [[101.0], [201.0]],
            "ask_sz": [[1.0], [1.0]],
        }
    )
    with pytest.raises(ValueError):
        build_feature_table(df)


def test_build_feature_table_runs_end_to_end_on_synthetic_data():
    n = 30
    rng = np.random.default_rng(0)
    base = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    df = pl.DataFrame(
        {
            "coin": ["BTC"] * n,
            "exch_time_ms": (np.arange(n) * 500).tolist(),
            "bid_px": [[p - 1.0, p - 2.0] for p in base],
            "bid_sz": [[1.0, 2.0] for _ in range(n)],
            "ask_px": [[p + 1.0, p + 2.0] for p in base],
            "ask_sz": [[1.0, 2.0] for _ in range(n)],
        }
    )
    out = build_feature_table(df, depth_levels=(2,))
    for col in (
        "spread_bps",
        "microprice",
        "tob_imbalance",
        "imbalance_d2",
        "ofi",
        "realized_vol",
        "momentum",
    ):
        assert col in out.columns
    assert out.height == n
