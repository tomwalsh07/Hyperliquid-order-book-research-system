import orjson
import polars as pl

from hlmicro.ingestion.normalize import normalize

L2BOOK_MSG = {
    "recv_ts_ns": 1_000_000_000,
    "channel": "l2Book",
    "data": {
        "coin": "BTC",
        "time": 1786636440652,
        "levels": [
            [{"px": "100.0", "sz": "1.0", "n": 2}, {"px": "99.0", "sz": "2.0", "n": 3}],
            [{"px": "101.0", "sz": "1.5", "n": 1}, {"px": "102.0", "sz": "2.5", "n": 4}],
        ],
    },
}
TRADES_MSG = {
    "recv_ts_ns": 1_000_000_500,
    "channel": "trades",
    "data": [
        {
            "coin": "BTC",
            "side": "B",
            "px": "100.5",
            "sz": "0.1",
            "time": 1786636440700,
            "hash": "0xabc",
            "tid": 1,
        },
        {
            "coin": "BTC",
            "side": "A",
            "px": "100.4",
            "sz": "0.2",
            "time": 1786636440800,
            "hash": "0xdef",
            "tid": 2,
        },
    ],
}
ASSET_CTX_MSG = {
    "recv_ts_ns": 1_000_001_000,
    "channel": "activeAssetCtx",
    "data": {
        "coin": "BTC",
        "ctx": {
            "funding": "0.0000125",
            "openInterest": "1000.0",
            "prevDayPx": "99.0",
            "dayNtlVlm": "500000.0",
            "premium": "0.0001",
            "oraclePx": "100.2",
            "markPx": "100.1",
            "midPx": "100.5",
            "dayBaseVlm": "5000.0",
        },
    },
}
PREDICTED_FUNDING_MSG = {
    "recv_ts_ns": 1_000_002_000,
    "channel": "predictedFundingsPoll",
    "data": [
        [
            "BTC",
            [
                [
                    "HlPerp",
                    {"fundingRate": "0.0000125", "nextFundingTime": 123, "fundingIntervalHours": 1},
                ]
            ],
        ]
    ],
}


def _write_raw_hour(raw_dir, date_str, hour_str, msgs):
    d = raw_dir / date_str
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{hour_str}.jsonl"
    with open(path, "ab") as f:
        for m in msgs:
            f.write(orjson.dumps(m))
            f.write(b"\n")
    return path


def test_normalize_produces_typed_parquet_per_table(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    _write_raw_hour(
        raw_dir, "20260813", "15", [L2BOOK_MSG, TRADES_MSG, ASSET_CTX_MSG, PREDICTED_FUNDING_MSG]
    )

    rows = normalize(raw_dir, out_dir)
    assert rows == {"l2book": 1, "trades": 2, "asset_ctx": 1, "predicted_funding": 1}

    l2 = pl.read_parquet(out_dir / "l2book" / "date=20260813" / "hour=15.parquet")
    assert l2.row(0, named=True)["bid_px"] == [100.0, 99.0]
    assert l2.row(0, named=True)["ask_px"] == [101.0, 102.0]

    trades = pl.read_parquet(out_dir / "trades" / "date=20260813" / "hour=15.parquet")
    assert trades["side"].to_list() == ["B", "A"]
    assert trades["tid"].to_list() == [1, 2]

    ctx = pl.read_parquet(out_dir / "asset_ctx" / "date=20260813" / "hour=15.parquet")
    assert ctx.row(0, named=True)["funding"] == 0.0000125

    pf = pl.read_parquet(out_dir / "predicted_funding" / "date=20260813" / "hour=15.parquet")
    assert pf.row(0, named=True)["venue"] == "HlPerp"


def test_normalize_is_idempotent_and_skips_unchanged_files(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    _write_raw_hour(raw_dir, "20260813", "15", [L2BOOK_MSG])

    rows1 = normalize(raw_dir, out_dir)
    assert rows1["l2book"] == 1

    rows2 = normalize(raw_dir, out_dir)  # nothing changed -> nothing reprocessed
    assert rows2 == dict.fromkeys(rows2, 0)


def test_normalize_reprocesses_growing_file_without_duplicating(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    _write_raw_hour(raw_dir, "20260813", "15", [L2BOOK_MSG])
    normalize(raw_dir, out_dir)

    # simulate the live collector appending more to the still-open hour file
    _write_raw_hour(raw_dir, "20260813", "15", [L2BOOK_MSG])
    normalize(raw_dir, out_dir)

    l2 = pl.read_parquet(out_dir / "l2book" / "date=20260813" / "hour=15.parquet")
    assert l2.shape[0] == 2  # rewritten from the full (now 2-line) raw file, not appended-to-twice
