"""Typed records for normalized (post-parse) market data.

These are the second-pass, validated shapes produced by
`hlmicro.ingestion.normalize` from raw captured WS messages. The raw capture
itself is untyped JSON (see `recorder.py`) — nothing here runs on the hot
ingestion path, only during normalization and downstream analytics.

Field semantics are documented in `docs/api_notes.md`, confirmed against a
live probe of the Hyperliquid WS API and the official docs on 2026-08-13.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class L2Level(BaseModel):
    px: float
    sz: float
    n: int


class L2BookSnapshot(BaseModel):
    """A full order-book replacement (see docs/api_notes.md §1 — Hyperliquid's
    l2Book channel pushes complete snapshots, never diffs)."""

    coin: str
    exch_time_ms: int
    recv_ts_ns: int
    bids: list[L2Level]
    asks: list[L2Level]


class Trade(BaseModel):
    coin: str
    side: Literal["B", "A"]  # B = taker bought (hit ask), A = taker sold (hit bid)
    px: float
    sz: float
    exch_time_ms: int
    recv_ts_ns: int
    tid: int
    trade_hash: str = Field(alias="hash")

    model_config = {"populate_by_name": True}


class AssetCtx(BaseModel):
    """Market-wide context for one symbol: current funding, mark/oracle/mid,
    open interest. Pushed continuously via the activeAssetCtx subscription."""

    coin: str
    recv_ts_ns: int
    funding: float
    open_interest: float
    prev_day_px: float
    day_ntl_vlm: float
    premium: float
    oracle_px: float
    mark_px: float
    mid_px: float
    day_base_vlm: float


class PredictedFunding(BaseModel):
    """Predicted next funding rate per venue, polled from the REST
    `predictedFundings` info endpoint (not a WS subscription). Hyperliquid's
    own venue is "HlPerp" (fundingIntervalHours=1); other venues (BinPerp,
    BybitPerp, ...) are included for context but settle on 8h cycles."""

    coin: str
    venue: str
    funding_rate: float
    next_funding_time_ms: int
    funding_interval_hours: int
    recv_ts_ns: int
