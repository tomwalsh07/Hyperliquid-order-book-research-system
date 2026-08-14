"""Periodic REST polling for data not available over the WS subscriptions.

Predicted funding rates are only exposed via the `predictedFundings` info
endpoint (POST /info), not a WS channel — see docs/api_notes.md §4. We poll
it on an interval and feed results through the same raw-first recorder as
everything else, tagged with a synthetic channel name so normalize.py can
tell it apart from genuine WS traffic.
"""

from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

INFO_URL = "https://api.hyperliquid.xyz/info"
PREDICTED_FUNDING_CHANNEL = "predictedFundingsPoll"


async def poll_predicted_fundings(
    symbols: list[str],
    recorder,
    stop_event: asyncio.Event,
    interval_s: float = 60.0,
) -> None:
    symbol_set = set(symbols)
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                async with session.post(INFO_URL, json={"type": "predictedFundings"}) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                recv_ts_ns = time.time_ns()
                filtered = [entry for entry in data if entry and entry[0] in symbol_set]
                recorder.write({"channel": PREDICTED_FUNDING_CHANNEL, "data": filtered}, recv_ts_ns)
            except Exception as exc:  # noqa: BLE001 - never let polling kill the collector
                logger.warning("predictedFundings poll failed: %r", exc)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            except TimeoutError:
                pass
