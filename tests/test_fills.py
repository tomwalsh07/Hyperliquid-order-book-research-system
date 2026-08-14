from hlmicro.backtest.fills import FillSimulator


def test_no_fill_below_consumption_threshold():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    # trade of 4 units < 50% of 10 -> no fill yet
    fills = sim.on_trade(trade_side="A", trade_price=100.0, trade_size=4.0)
    assert fills == {}


def test_fill_credited_once_cumulative_consumption_crosses_threshold():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    assert sim.on_trade("A", 100.0, 4.0) == {}  # cumulative 4 < 5
    fills = sim.on_trade("A", 100.0, 3.0)  # cumulative 7 >= 5 -> fill for 7-5=2
    assert fills == {"bid": 2.0}


def test_consumption_resets_after_a_credited_fill():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    fill1 = sim.on_trade("A", 100.0, 6.0)  # crosses threshold (5), fill=1, resets to 0
    assert fill1 == {"bid": 1.0}

    # next trade starts fresh accumulation from zero, needs to reach 5 again
    assert sim.on_trade("A", 100.0, 4.0) == {}  # 4 < 5, no fill yet
    fill2 = sim.on_trade("A", 100.0, 2.0)  # cumulative 6 >= 5 -> fill for 6-5=1
    assert fill2 == {"bid": 1.0}


def test_exact_threshold_crossing_credits_zero_or_positive_but_never_negative():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    fills = sim.on_trade("A", 100.0, 5.0)  # exactly the threshold
    assert fills.get("bid", 0.0) >= 0.0


def test_wrong_trade_side_does_not_fill_bid_quote():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    # "B" (taker bought, hits ASK) should never fill our BID
    fills = sim.on_trade("B", 100.0, 20.0)
    assert fills == {}


def test_wrong_price_does_not_fill():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    fills = sim.on_trade("A", 99.5, 20.0)  # right side, wrong price
    assert fills == {}


def test_unknown_zero_resting_depth_never_fills_conservatively():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=0.0)  # depth unknown/zero
    fills = sim.on_trade(
        "A", 100.0, 1000.0
    )  # huge print, but no visibility -> conservative no-fill
    assert fills == {}


def test_moving_quote_price_resets_consumption_counter():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    sim.on_trade("A", 100.0, 4.0)  # 4/5 toward threshold at 100.0

    sim.set_quote("bid", price=99.0, resting_depth_at_price=10.0)  # requote to a new price
    # old accumulation at 100.0 must not carry over to 99.0
    fills = sim.on_trade("A", 99.0, 4.0)
    assert fills == {}


def test_bid_and_ask_consumption_tracked_independently():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    sim.set_quote("ask", price=101.0, resting_depth_at_price=10.0)

    # a sell-side print hits our bid only
    fills = sim.on_trade("A", 100.0, 6.0)
    assert fills == {"bid": 1.0}

    # a buy-side print at the ask should not have been affected by the above
    fills2 = sim.on_trade("B", 101.0, 6.0)
    assert fills2 == {"ask": 1.0}


def test_withdrawing_quote_stops_future_fills():
    sim = FillSimulator(min_consumption_pct=0.5)
    sim.set_quote("bid", price=100.0, resting_depth_at_price=10.0)
    sim.set_quote("bid", price=None, resting_depth_at_price=None)  # kill-switch pulls the quote
    fills = sim.on_trade("A", 100.0, 100.0)
    assert fills == {}
