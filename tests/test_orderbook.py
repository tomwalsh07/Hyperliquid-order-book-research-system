from hlmicro.orderbook.book import OrderBook


def test_load_snapshot_sets_book_state():
    book = OrderBook(symbol="BTC")
    book.load_snapshot(
        bid_px=[100.0, 99.0, 98.0],
        bid_sz=[1.0, 2.0, 3.0],
        ask_px=[101.0, 102.0, 103.0],
        ask_sz=[1.5, 2.5, 3.5],
        exch_time_ms=1000,
    )
    assert book.best_bid() == (100.0, 1.0)
    assert book.best_ask() == (101.0, 1.5)
    assert book.mid_price() == 100.5
    assert book.last_update_ms == 1000
    assert not book.is_crossed()


def test_snapshot_fully_replaces_prior_state():
    """A stale price level from an earlier snapshot must not survive a
    fresh load_snapshot -- this is the "reset" semantics l2Book relies on."""
    book = OrderBook(symbol="BTC")
    book.load_snapshot([100.0], [1.0], [101.0], [1.0], exch_time_ms=1000)
    assert 100.0 in book.bids

    book.load_snapshot([105.0], [2.0], [106.0], [2.0], exch_time_ms=2000)
    assert 100.0 not in book.bids
    assert book.best_bid() == (105.0, 2.0)
    assert book.best_ask() == (106.0, 2.0)


def test_apply_level_update_upsert():
    book = OrderBook(symbol="BTC")
    book.load_snapshot([100.0], [1.0], [101.0], [1.0], exch_time_ms=1000)

    book.apply_level_update("bid", price=99.0, size=5.0, n=3)
    assert book.bids[99.0] == 5.0
    assert book.bid_n[99.0] == 3

    # update an existing level's size
    book.apply_level_update("bid", price=100.0, size=10.0)
    assert book.bids[100.0] == 10.0
    assert book.best_bid() == (100.0, 10.0)


def test_apply_level_update_delete_on_zero_size():
    book = OrderBook(symbol="BTC")
    book.load_snapshot(
        bid_px=[100.0, 99.0], bid_sz=[1.0, 2.0], ask_px=[101.0], ask_sz=[1.0], exch_time_ms=1000
    )
    book.apply_level_update("bid", price=100.0, size=0.0)
    assert 100.0 not in book.bids
    assert book.best_bid() == (99.0, 2.0)


def test_crossed_book_detection():
    book = OrderBook(symbol="BTC")
    book.load_snapshot([100.0], [1.0], [101.0], [1.0], exch_time_ms=1000)
    assert not book.is_crossed()

    # simulate a bad update that crosses the book (best bid >= best ask)
    book.apply_level_update("bid", price=101.5, size=1.0)
    assert book.is_crossed()


def test_empty_book_has_no_best_bid_ask_or_mid():
    book = OrderBook(symbol="BTC")
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.mid_price() is None
    assert not book.is_crossed()


def test_bid_ask_levels_sorted_correctly_and_truncated():
    book = OrderBook(symbol="BTC")
    book.load_snapshot(
        bid_px=[98.0, 100.0, 99.0],
        bid_sz=[3.0, 1.0, 2.0],
        ask_px=[103.0, 101.0, 102.0],
        ask_sz=[3.0, 1.0, 2.0],
        exch_time_ms=1000,
    )
    assert book.bid_levels() == [(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)]
    assert book.ask_levels() == [(101.0, 1.0), (102.0, 2.0), (103.0, 3.0)]
    assert book.bid_levels(n=2) == [(100.0, 1.0), (99.0, 2.0)]


def test_reset_clears_all_state():
    book = OrderBook(symbol="BTC")
    book.load_snapshot([100.0], [1.0], [101.0], [1.0], exch_time_ms=1000, bid_n=[2], ask_n=[3])
    book.reset()
    assert book.bids == {} and book.asks == {}
    assert book.bid_n == {} and book.ask_n == {}
    assert book.last_update_ms is None
