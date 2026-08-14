import numpy as np
import polars as pl
import pytest

from hlmicro.research.stats import (
    chronological_split,
    directional_accuracy,
    hac_regression,
    information_coefficient,
    net_of_cost_check,
    newey_west_maxlags,
)

# ---- chronological_split ---------------------------------------------------


def test_chronological_split_uses_time_not_row_count():
    # uneven cadence: most rows clustered early, a few late
    times = [0, 10, 20, 30, 40, 1000]
    df = pl.DataFrame({"exch_time_ms": times, "v": list(range(6))})
    train, test = chronological_split(df, train_fraction=0.5)
    # cutoff = 0 + 0.5*(1000-0) = 500 -> only the last row (t=1000) is test
    assert train["exch_time_ms"].to_list() == [0, 10, 20, 30, 40]
    assert test["exch_time_ms"].to_list() == [1000]


def test_chronological_split_never_reorders_or_leaks():
    df = pl.DataFrame({"exch_time_ms": [0, 100, 200, 300, 400], "v": [1, 2, 3, 4, 5]})
    train, test = chronological_split(df, train_fraction=0.6)
    assert train["exch_time_ms"].max() <= test["exch_time_ms"].min()


# ---- information_coefficient -----------------------------------------------


def test_information_coefficient_perfect_positive_correlation():
    feature = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    label = feature * 2.0
    res = information_coefficient(feature, label)
    assert res["pearson"] == pytest.approx(1.0)
    assert res["spearman"] == pytest.approx(1.0)
    assert res["n"] == 10


def test_information_coefficient_handles_nan_and_small_n():
    feature = np.array([1.0, np.nan, 3.0])
    label = np.array([1.0, 2.0, np.nan])
    res = information_coefficient(feature, label)
    assert res["n"] == 1  # only index 0 has both non-nan
    assert np.isnan(res["pearson"])  # too few points (<10) -> nan by design


# ---- directional_accuracy --------------------------------------------------


def test_directional_accuracy_hand_computed():
    # 4 matches, 1 mismatch out of 5 -> hit_rate = 0.8
    feature = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    label = np.array([0.5, -0.3, 0.2, 0.1, -0.4])  # signs: +,-,+,+,- -> matches: T,T,T,F,F = 3/5
    res = directional_accuracy(feature, label)
    assert res["n"] == 5
    assert res["hit_rate"] == pytest.approx(3 / 5)


def test_directional_accuracy_excludes_zeros():
    feature = np.array([1.0, 0.0, -1.0])
    label = np.array([0.5, 0.2, 0.0])
    res = directional_accuracy(feature, label)
    assert res["n"] == 1  # only index 0 has nonzero feature AND label


def test_directional_accuracy_binomial_pvalue_significant_for_strong_edge():
    rng = np.random.default_rng(1)
    feature = rng.choice([-1.0, 1.0], size=200)
    label = feature * np.abs(rng.normal(1, 0.1, 200))  # perfectly aligned sign
    res = directional_accuracy(feature, label)
    assert res["hit_rate"] == pytest.approx(1.0)
    assert res["p_value"] < 0.001


# ---- newey_west_maxlags -----------------------------------------------------


def test_newey_west_maxlags_scales_with_horizon_over_sampling_interval():
    # horizon 5000ms, sampling every 250ms -> ~20 lags
    assert newey_west_maxlags(5000, 250) == 20
    # horizon shorter than sampling interval -> floors at 1
    assert newey_west_maxlags(100, 250) == 1


def test_newey_west_maxlags_handles_zero_sampling_interval():
    assert newey_west_maxlags(1000, 0) == 1


# ---- hac_regression ----------------------------------------------------


def test_hac_regression_recovers_known_linear_relationship():
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(0, 1, n)
    y = 0.5 * x + rng.normal(0, 0.1, n)  # y ~ 0.5*x, low noise
    df = pl.DataFrame({"y": y, "x": x})
    res = hac_regression(df, y_col="y", x_cols=["x"], maxlags=5)
    assert res["params"]["x"] == pytest.approx(0.5, abs=0.05)
    assert res["pvalues"]["x"] < 0.001


def test_hac_regression_too_few_observations_returns_nan_gracefully():
    df = pl.DataFrame({"y": [1.0, 2.0], "x": [1.0, 2.0]})
    res = hac_regression(df, y_col="y", x_cols=["x"], maxlags=1)
    assert np.isnan(res["r_squared"])


# ---- net_of_cost_check -------------------------------------------------


def test_net_of_cost_check_hand_computed():
    # imbalance=0.5 (long signal), label (log ret) = 0.001 -> 10bps raw
    # spread=2bps, taker_fee=1bps -> cost = 2 + 2*1 = 4bps -> net = 6bps
    df = pl.DataFrame(
        {
            "imbalance": [0.5, -0.5],
            "label": [0.0010, -0.0010],  # both moves align with signal direction
            "spread_bps": [2.0, 2.0],
        }
    )
    res = net_of_cost_check(
        df,
        imbalance_col="imbalance",
        label_col="label",
        spread_bps_col="spread_bps",
        threshold=0.1,
        taker_fee_bps=1.0,
    )
    assert res["n"] == 2
    assert res["mean_net_bps"] == pytest.approx(10.0 - 4.0, abs=0.01)
    assert res["pct_positive"] == pytest.approx(1.0)


def test_net_of_cost_check_filters_below_threshold():
    df = pl.DataFrame({"imbalance": [0.05], "label": [0.01], "spread_bps": [1.0]})
    res = net_of_cost_check(
        df,
        imbalance_col="imbalance",
        label_col="label",
        spread_bps_col="spread_bps",
        threshold=0.1,
        taker_fee_bps=1.0,
    )
    assert res["n"] == 0


def test_net_of_cost_check_empty_after_filter_does_not_crash():
    df = pl.DataFrame(
        schema={"imbalance": pl.Float64, "label": pl.Float64, "spread_bps": pl.Float64}
    )
    res = net_of_cost_check(
        df,
        imbalance_col="imbalance",
        label_col="label",
        spread_bps_col="spread_bps",
        threshold=0.1,
        taker_fee_bps=1.0,
    )
    assert res["n"] == 0
