from hlmicro.backtest.costs import maker_fee, taker_fee


def test_maker_fee_hand_computed():
    # 1000 notional * 1.5bps = 1000 * 0.00015 = 0.15
    assert maker_fee(1000.0, 1.5) == 0.15


def test_taker_fee_hand_computed():
    # 1000 * 4.5bps = 0.45
    assert taker_fee(1000.0, 4.5) == 0.45


def test_fees_use_absolute_notional():
    assert maker_fee(-1000.0, 1.5) == 0.15
