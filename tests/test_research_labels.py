import numpy as np
import polars as pl
import pytest

from hlmicro.research.labels import add_forward_return_labels, forward_index_map

# ---- no-lookahead: the core integrity requirement ------------------------


def test_forward_index_map_never_points_at_or_before_current_row():
    """Direct assertion on the index mapping itself (not just output
    values): for every row i and horizon h, the mapped index must point to
    a STRICTLY later timestamp than row i's own timestamp."""
    times = np.array([0, 100, 250, 400, 900, 1500, 3000, 3050, 5000])
    for h in (100, 500, 1000, 5000, 10000):
        idx = forward_index_map(times, h)
        for i in range(len(times)):
            if idx[i] >= len(times):
                continue  # no future observation exists - must yield null label, checked elsewhere
            assert idx[i] > i, f"h={h}, i={i}: mapped index {idx[i]} is not strictly after i"
            assert times[idx[i]] > times[i], f"h={h}, i={i}: mapped timestamp not strictly after t"
            assert times[idx[i]] >= times[i] + h


def test_forward_index_map_picks_earliest_qualifying_future_point():
    # t=0, h=250 -> first time >= 250 is index 2 (t=250), not index 3 (t=400)
    times = np.array([0, 100, 250, 400, 900])
    idx = forward_index_map(times, 250)
    assert idx[0] == 2


def test_forward_index_map_out_of_range_at_series_end():
    times = np.array([0, 100, 200])
    idx = forward_index_map(times, 1000)
    assert idx[-1] >= len(times)  # no future observation exists -> caller must null this out


def test_labels_never_use_a_value_from_the_row_itself_or_earlier():
    """Construct a series where every price is a unique, identifiable
    marker (its own index), then verify no label could possibly have been
    computed from index <= i by checking the underlying index map, and
    cross-check the returned label matches the value at a later index."""
    n = 20
    times = np.arange(n) * 200  # 200ms apart
    # prices are strictly increasing so we can identify exactly which
    # index produced a given forward price
    prices = 100.0 + np.arange(n, dtype=float)
    df = pl.DataFrame(
        {"coin": ["BTC"] * n, "exch_time_ms": times, "mid": prices, "microprice": prices}
    )

    out = add_forward_return_labels(df, horizons_ms=[500], price_cols=("mid",))
    log_ret = out["fwd_logret_mid_500ms"].to_numpy()

    idx_map = forward_index_map(times, 500)
    for i in range(n):
        if idx_map[i] >= n:
            assert np.isnan(log_ret[i])
            continue
        expected = np.log(prices[idx_map[i]]) - np.log(prices[i])
        assert log_ret[i] == pytest.approx(expected)
        assert idx_map[i] > i  # re-assert strictly-future for every non-null label


# ---- correctness -----------------------------------------------------------


def test_add_forward_return_labels_hand_computed():
    times = np.array([0, 100, 200, 300])
    mids = np.array([100.0, 101.0, 102.0, 104.0])
    df = pl.DataFrame({"coin": ["BTC"] * 4, "exch_time_ms": times, "mid": mids, "microprice": mids})

    out = add_forward_return_labels(df, horizons_ms=[100], price_cols=("mid",))
    # row0 (t=0): first t>=100 is row1 (mid=101) -> log(101/100)
    assert out["fwd_logret_mid_100ms"][0] == pytest.approx(np.log(101.0 / 100.0))
    # row2 (t=200): first t>=300 is row3 (mid=104) -> log(104/102)
    assert out["fwd_logret_mid_100ms"][2] == pytest.approx(np.log(104.0 / 102.0))
    # row3 (t=300): no future row -> null
    assert out["fwd_logret_mid_100ms"][3] is None or np.isnan(out["fwd_logret_mid_100ms"][3])


def test_add_forward_return_labels_multiple_horizons_and_price_cols():
    times = np.array([0, 100, 500, 1000])
    mid = np.array([100.0, 100.0, 110.0, 121.0])
    micro = np.array([100.0, 100.5, 109.0, 120.0])
    df = pl.DataFrame({"coin": ["ETH"] * 4, "exch_time_ms": times, "mid": mid, "microprice": micro})
    out = add_forward_return_labels(df, horizons_ms=[100, 1000], price_cols=("mid", "microprice"))
    assert "fwd_logret_mid_100ms" in out.columns
    assert "fwd_logret_microprice_100ms" in out.columns
    assert "fwd_logret_mid_1000ms" in out.columns
    assert "fwd_logret_microprice_1000ms" in out.columns
    # row0, h=1000 -> first t>=1000 is row3 (mid=121)
    assert out["fwd_logret_mid_1000ms"][0] == pytest.approx(np.log(121.0 / 100.0))


def test_tail_labels_are_real_polars_nulls_not_float_nan():
    """Regression test: a float NaN is a normal value to polars and
    silently survives .drop_nulls(), which previously let "missing
    horizon" rows leak into every downstream stat as poison NaN values
    instead of being excluded. The tail label must be a proper null."""
    times = np.array([0, 100, 200])
    mids = np.array([100.0, 101.0, 102.0])
    df = pl.DataFrame({"coin": ["BTC"] * 3, "exch_time_ms": times, "mid": mids, "microprice": mids})
    out = add_forward_return_labels(df, horizons_ms=[1000], price_cols=("mid",))
    assert out["fwd_logret_mid_1000ms"].null_count() == 3
    assert out.drop_nulls().height == 0


def test_multi_symbol_input_is_rejected():
    df = pl.DataFrame(
        {
            "coin": ["BTC", "ETH"],
            "exch_time_ms": [0, 100],
            "mid": [100.0, 200.0],
            "microprice": [100.0, 200.0],
        }
    )
    with pytest.raises(ValueError):
        add_forward_return_labels(df, horizons_ms=[100])
