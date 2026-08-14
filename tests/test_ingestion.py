import orjson

from hlmicro.ingestion.recorder import RawRecorder
from hlmicro.ingestion.ws_client import HyperliquidWSClient


def _ns(date_str: str, hour_str: str, minute: int = 0) -> int:
    from datetime import UTC, datetime

    dt = datetime.strptime(f"{date_str}{hour_str}{minute:02d}", "%Y%m%d%H%M").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1e9)


def test_recorder_writes_and_rotates_by_hour(tmp_path):
    rec = RawRecorder(tmp_path, flush_every_n=1)
    rec.write({"channel": "l2Book", "data": {"coin": "BTC"}}, _ns("20260813", "15", 0))
    rec.write({"channel": "l2Book", "data": {"coin": "BTC"}}, _ns("20260813", "15", 30))
    rec.write({"channel": "l2Book", "data": {"coin": "BTC"}}, _ns("20260813", "16", 0))
    rec.close()

    hour15 = tmp_path / "20260813" / "15.jsonl"
    hour16 = tmp_path / "20260813" / "16.jsonl"
    assert hour15.exists() and hour16.exists()
    assert len(hour15.read_text().strip().splitlines()) == 2
    assert len(hour16.read_text().strip().splitlines()) == 1


def test_recorder_restart_appends_without_truncating(tmp_path):
    rec1 = RawRecorder(tmp_path, flush_every_n=1)
    rec1.write({"channel": "l2Book", "data": {"coin": "BTC"}}, _ns("20260813", "15", 0))
    rec1.close()

    # Simulate a fresh process restarting into the same output dir.
    rec2 = RawRecorder(tmp_path, flush_every_n=1)
    rec2.write({"channel": "l2Book", "data": {"coin": "BTC"}}, _ns("20260813", "15", 1))
    rec2.close()

    lines = (tmp_path / "20260813" / "15.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_recorder_preserves_message_content_and_injects_recv_ts(tmp_path):
    rec = RawRecorder(tmp_path, flush_every_n=1)
    msg = {"channel": "trades", "data": [{"coin": "ETH", "px": "3000.0"}]}
    ts = _ns("20260813", "15", 0)
    rec.write(msg, ts)
    rec.close()

    line = (tmp_path / "20260813" / "15.jsonl").read_text().strip()
    parsed = orjson.loads(line)
    assert parsed["recv_ts_ns"] == ts
    assert parsed["channel"] == "trades"
    assert parsed["data"] == msg["data"]


def test_gap_detection_flags_large_time_deltas():
    client = HyperliquidWSClient(symbols=["BTC"], on_message=None, gap_warn_threshold_s=15.0)
    last_seen: dict = {}

    client._check_gap({"channel": "l2Book", "data": {"coin": "BTC", "time": 1_000_000}}, last_seen)
    assert client.stats.gaps_detected == 0

    # +5s: within normal cadence, no warning
    client._check_gap({"channel": "l2Book", "data": {"coin": "BTC", "time": 1_005_000}}, last_seen)
    assert client.stats.gaps_detected == 0

    # +20s: exceeds threshold
    client._check_gap({"channel": "l2Book", "data": {"coin": "BTC", "time": 1_025_000}}, last_seen)
    assert client.stats.gaps_detected == 1


def test_gap_detection_ignores_list_shaped_trades_payload():
    client = HyperliquidWSClient(symbols=["BTC"], on_message=None)
    last_seen: dict = {}
    # trades payloads are lists, not dicts -> must not raise or count as a gap source
    client._check_gap(
        {"channel": "trades", "data": [{"coin": "BTC", "time": 1_000_000}]}, last_seen
    )
    assert client.stats.gaps_detected == 0
    assert last_seen == {}
