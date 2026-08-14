"""Second-pass normalization: raw JSONL capture -> typed Parquet.

Idempotent and incremental: each raw hour-file is tracked in a manifest by
(size, mtime). If a raw file hasn't changed since last run, it's skipped.
If it has changed (e.g. it's the currently-open hour, still being appended
by a live collector), the corresponding output partition is fully
rewritten from that raw file — never appended-to twice, so re-running
normalization (including after a restart) can never duplicate rows.

One row per book snapshot event is kept as list-columns (bid/ask px/sz/n
arrays) rather than exploded into one row per level: this preserves the
ladder structure needed for depth-weighted analytics and is far cheaper to
store than a 20x row explosion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import orjson
import polars as pl

logger = logging.getLogger(__name__)

MANIFEST_NAME = "_normalize_manifest.json"

L2BOOK_SCHEMA = {
    "coin": pl.Utf8,
    "exch_time_ms": pl.Int64,
    "recv_ts_ns": pl.Int64,
    "bid_px": pl.List(pl.Float64),
    "bid_sz": pl.List(pl.Float64),
    "bid_n": pl.List(pl.Int64),
    "ask_px": pl.List(pl.Float64),
    "ask_sz": pl.List(pl.Float64),
    "ask_n": pl.List(pl.Int64),
}
TRADES_SCHEMA = {
    "coin": pl.Utf8,
    "side": pl.Utf8,
    "px": pl.Float64,
    "sz": pl.Float64,
    "exch_time_ms": pl.Int64,
    "recv_ts_ns": pl.Int64,
    "tid": pl.Int64,
    "trade_hash": pl.Utf8,
}
ASSET_CTX_SCHEMA = {
    "coin": pl.Utf8,
    "recv_ts_ns": pl.Int64,
    "funding": pl.Float64,
    "open_interest": pl.Float64,
    "prev_day_px": pl.Float64,
    "day_ntl_vlm": pl.Float64,
    "premium": pl.Float64,
    "oracle_px": pl.Float64,
    "mark_px": pl.Float64,
    "mid_px": pl.Float64,
    "day_base_vlm": pl.Float64,
}
PREDICTED_FUNDING_SCHEMA = {
    "coin": pl.Utf8,
    "venue": pl.Utf8,
    "funding_rate": pl.Float64,
    "next_funding_time_ms": pl.Int64,
    "funding_interval_hours": pl.Int64,
    "recv_ts_ns": pl.Int64,
}

TABLES = ("l2book", "trades", "asset_ctx", "predicted_funding")


def parse_raw_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "l2Book": [],
        "trades": [],
        "activeAssetCtx": [],
        "predictedFundingsPoll": [],
    }
    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = orjson.loads(line)
            except orjson.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", path)
                continue
            channel = msg.get("channel")
            if channel in buckets:
                buckets[channel].append(msg)
    return buckets


def _l2book_rows(msgs: list[dict]) -> list[dict]:
    rows = []
    for m in msgs:
        d = m["data"]
        bids, asks = d["levels"]
        rows.append(
            {
                "coin": d["coin"],
                "exch_time_ms": d["time"],
                "recv_ts_ns": m["recv_ts_ns"],
                "bid_px": [float(lv["px"]) for lv in bids],
                "bid_sz": [float(lv["sz"]) for lv in bids],
                "bid_n": [int(lv["n"]) for lv in bids],
                "ask_px": [float(lv["px"]) for lv in asks],
                "ask_sz": [float(lv["sz"]) for lv in asks],
                "ask_n": [int(lv["n"]) for lv in asks],
            }
        )
    return rows


def _trades_rows(msgs: list[dict]) -> list[dict]:
    rows = []
    for m in msgs:
        for t in m["data"]:
            rows.append(
                {
                    "coin": t["coin"],
                    "side": t["side"],
                    "px": float(t["px"]),
                    "sz": float(t["sz"]),
                    "exch_time_ms": t["time"],
                    "recv_ts_ns": m["recv_ts_ns"],
                    "tid": int(t["tid"]),
                    "trade_hash": t["hash"],
                }
            )
    return rows


def _asset_ctx_rows(msgs: list[dict]) -> list[dict]:
    rows = []
    for m in msgs:
        d = m["data"]
        ctx = d["ctx"]
        rows.append(
            {
                "coin": d["coin"],
                "recv_ts_ns": m["recv_ts_ns"],
                "funding": float(ctx["funding"]),
                "open_interest": float(ctx["openInterest"]),
                "prev_day_px": float(ctx["prevDayPx"]),
                "day_ntl_vlm": float(ctx["dayNtlVlm"]),
                "premium": float(ctx["premium"]),
                "oracle_px": float(ctx["oraclePx"]),
                "mark_px": float(ctx["markPx"]),
                "mid_px": float(ctx["midPx"]),
                "day_base_vlm": float(ctx["dayBaseVlm"]),
            }
        )
    return rows


def _predicted_funding_rows(msgs: list[dict]) -> list[dict]:
    rows = []
    for m in msgs:
        for coin, venues in m["data"]:
            for venue_name, info in venues:
                if not info:
                    continue
                rows.append(
                    {
                        "coin": coin,
                        "venue": venue_name,
                        "funding_rate": float(info["fundingRate"]),
                        "next_funding_time_ms": int(info["nextFundingTime"]),
                        "funding_interval_hours": int(info.get("fundingIntervalHours", 0)),
                        "recv_ts_ns": m["recv_ts_ns"],
                    }
                )
    return rows


_ROW_BUILDERS = {
    "l2book": ("l2Book", _l2book_rows, L2BOOK_SCHEMA),
    "trades": ("trades", _trades_rows, TRADES_SCHEMA),
    "asset_ctx": ("activeAssetCtx", _asset_ctx_rows, ASSET_CTX_SCHEMA),
    "predicted_funding": (
        "predictedFundingsPoll",
        _predicted_funding_rows,
        PREDICTED_FUNDING_SCHEMA,
    ),
}


def _load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


def normalize(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    """Returns a dict of table -> rows written in this run (changed files only)."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)
    rows_written = dict.fromkeys(TABLES, 0)

    raw_files = sorted(raw_dir.glob("*/*.jsonl"))
    for raw_path in raw_files:
        rel_key = str(raw_path.relative_to(raw_dir)).replace("\\", "/")
        stat = raw_path.stat()
        fingerprint = {"size": stat.st_size, "mtime": stat.st_mtime}
        if manifest.get(rel_key) == fingerprint:
            continue  # unchanged since last run

        date_str, hour_file = rel_key.split("/")
        hour_str = hour_file.removesuffix(".jsonl")
        buckets = parse_raw_file(raw_path)

        for table, (channel, builder, schema) in _ROW_BUILDERS.items():
            rows = builder(buckets[channel])
            table_dir = out_dir / table / f"date={date_str}"
            table_dir.mkdir(parents=True, exist_ok=True)
            out_path = table_dir / f"hour={hour_str}.parquet"
            if rows:
                df = pl.DataFrame(rows, schema=schema)
                df.write_parquet(out_path)
                rows_written[table] += len(rows)
            elif out_path.exists():
                out_path.unlink()  # e.g. no trades printed in a quiet hour

        manifest[rel_key] = fingerprint
        logger.info("Normalized %s", rel_key)

    _save_manifest(out_dir, manifest)
    return rows_written
