from hlmicro.strategies.base import Fill, MarketState
from hlmicro.strategies.inventory_mm import InventoryAwareMM
from hlmicro.strategies.naive_mm import NaiveSymmetricMM


def _state(mid=100.0, micro=100.0, ts=1000, drought=False) -> MarketState:
    return MarketState(
        coin="BTC",
        timestamp_ms=ts,
        best_bid=mid - 0.5,
        best_bid_sz=1.0,
        best_ask=mid + 0.5,
        best_ask_sz=1.0,
        mid=mid,
        microprice=micro,
        tob_imbalance=0.0,
        liquidity_drought=drought,
    )


# ---- NaiveSymmetricMM -------------------------------------------------


def test_naive_mm_quotes_symmetric_around_microprice():
    strat = NaiveSymmetricMM(half_spread_bps=100.0, quote_size=1.0, max_inventory=5.0)
    q = strat.get_quotes(_state(micro=100.0))
    # half = 100 * 100bps/10000 = 1.0
    assert q.bid_px == 99.0
    assert q.ask_px == 101.0
    assert q.bid_sz == 1.0 and q.ask_sz == 1.0


def test_naive_mm_pulls_quotes_during_liquidity_drought():
    strat = NaiveSymmetricMM(half_spread_bps=10.0, quote_size=1.0, max_inventory=5.0)
    q = strat.get_quotes(_state(drought=True))
    assert q.bid_sz == 0.0 and q.ask_sz == 0.0


def test_naive_mm_respects_max_inventory_per_side():
    strat = NaiveSymmetricMM(half_spread_bps=10.0, quote_size=1.0, max_inventory=1.0)
    strat.on_fill(Fill(timestamp_ms=1, side="buy", price=100.0, size=1.0))
    assert strat.inventory == 1.0
    q = strat.get_quotes(_state())
    assert q.bid_sz == 0.0  # already at max long, stop buying more
    assert q.ask_sz == 1.0  # still free to sell (reduce inventory)


def test_naive_mm_on_fill_updates_inventory_signed():
    strat = NaiveSymmetricMM(half_spread_bps=10.0, quote_size=1.0, max_inventory=5.0)
    strat.on_fill(Fill(timestamp_ms=1, side="buy", price=100.0, size=2.0))
    strat.on_fill(Fill(timestamp_ms=2, side="sell", price=100.0, size=0.5))
    assert strat.inventory == 1.5


# ---- InventoryAwareMM ----------------------------------------------------


def test_inventory_mm_flat_quotes_symmetric_around_mid():
    strat = InventoryAwareMM(
        risk_aversion=0.0001,
        vol_lookback_s=60,
        decay_window_s=300,
        max_inventory=5.0,
        quote_size=1.0,
        base_half_spread_bps=10.0,
    )
    state = _state(mid=100.0)
    strat.on_book_update(state)
    q = strat.get_quotes(state)
    # flat inventory, zero volatility history -> reservation price == mid,
    # half spread == base floor (100 * 10bps/10000 = 0.1)
    assert q.bid_px == 99.9
    assert q.ask_px == 100.1


def test_inventory_mm_skews_reservation_price_against_long_inventory():
    strat = InventoryAwareMM(
        risk_aversion=1.0,
        vol_lookback_s=60,
        decay_window_s=10,
        max_inventory=5.0,
        quote_size=1.0,
        base_half_spread_bps=1.0,
    )
    # feed some price history so sigma > 0
    for i, mid in enumerate([100.0, 101.0, 99.0, 102.0]):
        strat.on_book_update(_state(mid=mid, ts=1000 * i))

    strat.on_fill(Fill(timestamp_ms=1, side="buy", price=100.0, size=2.0))  # long inventory
    state = _state(mid=100.0, ts=5000)
    strat.on_book_update(state)
    q = strat.get_quotes(state)

    reservation = (q.bid_px + q.ask_px) / 2
    # positive inventory (long) should skew reservation price DOWN (below
    # mid) to encourage selling and discourage buying more
    assert reservation < state.mid


def test_inventory_mm_pulls_quotes_during_liquidity_drought():
    strat = InventoryAwareMM(
        risk_aversion=0.0001,
        vol_lookback_s=60,
        decay_window_s=300,
        max_inventory=5.0,
        quote_size=1.0,
    )
    q = strat.get_quotes(_state(drought=True))
    assert q.bid_sz == 0.0 and q.ask_sz == 0.0


def test_inventory_mm_respects_max_inventory():
    strat = InventoryAwareMM(
        risk_aversion=0.0001,
        vol_lookback_s=60,
        decay_window_s=300,
        max_inventory=1.0,
        quote_size=1.0,
    )
    strat.on_fill(Fill(timestamp_ms=1, side="sell", price=100.0, size=1.0))
    assert strat.inventory == -1.0
    q = strat.get_quotes(_state())
    assert q.ask_sz == 0.0  # already at max short, stop selling more
    assert q.bid_sz == 1.0  # still free to buy (reduce short)


def test_inventory_mm_vol_lookback_drops_old_history():
    strat = InventoryAwareMM(
        risk_aversion=0.0001,
        vol_lookback_s=10,
        decay_window_s=300,
        max_inventory=5.0,
        quote_size=1.0,
    )
    strat.on_book_update(_state(mid=100.0, ts=0))
    strat.on_book_update(_state(mid=200.0, ts=20_000))  # 20s later, > 10s lookback
    # the first point should have been dropped from history
    assert all(ts >= 20_000 - 10 * 1000 for ts, _ in strat._mid_history)
