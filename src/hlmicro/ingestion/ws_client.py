"""Async WebSocket client for Hyperliquid's public market-data feed.

Public market data only — no API keys, no account/auth. Subscribes per
symbol to l2Book, trades, and activeAssetCtx (see docs/api_notes.md for why
bbo is deliberately excluded and why l2Book needs no diff-continuity logic).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import websockets

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict, int], Awaitable[None]]


@dataclass
class WSClientStats:
    messages_received: int = 0
    reconnects: int = 0
    gaps_detected: int = 0
    last_error: str | None = None


@dataclass
class HyperliquidWSClient:
    symbols: list[str]
    on_message: OnMessage
    ws_url: str = "wss://api.hyperliquid.xyz/ws"
    ping_interval_s: float = 20.0
    reconnect_backoff_base_s: float = 1.0
    reconnect_backoff_max_s: float = 60.0
    reconnect_jitter_s: float = 1.0
    gap_warn_threshold_s: float = 15.0
    stats: WSClientStats = field(default_factory=WSClientStats)

    async def run(self, stop_event: asyncio.Event) -> None:
        """Connect-stream-reconnect loop. Runs until stop_event is set."""
        attempt = 0
        while not stop_event.is_set():
            try:
                await self._connect_and_stream(stop_event)
                attempt = 0  # clean exit (stop requested) or clean stream, reset backoff
            except Exception as exc:  # noqa: BLE001 - reconnect on anything, log it
                self.stats.last_error = repr(exc)
                logger.warning("WS disconnected (attempt %d): %r", attempt, exc)

            if stop_event.is_set():
                break

            self.stats.reconnects += 1
            delay = min(
                self.reconnect_backoff_base_s * (2**attempt), self.reconnect_backoff_max_s
            ) + random.uniform(0, self.reconnect_jitter_s)
            attempt += 1
            logger.info("Reconnecting in %.1fs (attempt %d)", delay, attempt)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def _connect_and_stream(self, stop_event: asyncio.Event) -> None:
        # Application-level ping/pong is the confirmed keepalive contract
        # (docs/api_notes.md §6); disable the library's own ping frames to
        # avoid two independent keepalive mechanisms racing each other.
        async with websockets.connect(self.ws_url, ping_interval=None) as ws:
            logger.info("WS connected: %s", self.ws_url)
            await self._subscribe_all(ws)

            ping_task = asyncio.create_task(self._ping_loop(ws, stop_event))
            last_exch_time_ms: dict[tuple[str, str], int] = {}
            try:
                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except TimeoutError:
                        continue
                    recv_ts_ns = time.time_ns()
                    msg = json.loads(raw)
                    self.stats.messages_received += 1

                    channel = msg.get("channel")
                    if channel == "pong":
                        continue
                    if channel == "subscriptionResponse":
                        logger.debug("Subscription ack: %s", msg["data"])
                        continue

                    self._check_gap(msg, last_exch_time_ms)
                    await self.on_message(msg, recv_ts_ns)
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    def _check_gap(self, msg: dict, last_exch_time_ms: dict[tuple[str, str], int]) -> None:
        """Flag suspicious gaps between consecutive dict-shaped payloads
        (l2Book / activeAssetCtx). Trades arrive as event lists with no
        implied cadence, so gaps there are expected and not tracked."""
        channel = msg.get("channel")
        data = msg.get("data")
        if not isinstance(data, dict):
            return
        coin = data.get("coin")
        exch_ms = data.get("time")
        if coin is None or exch_ms is None:
            return
        key = (channel, coin)
        prev = last_exch_time_ms.get(key)
        if prev is not None:
            gap_s = (exch_ms - prev) / 1000.0
            if gap_s > self.gap_warn_threshold_s:
                self.stats.gaps_detected += 1
                logger.warning("Gap detected on %s/%s: %.1fs between updates", channel, coin, gap_s)
        last_exch_time_ms[key] = exch_ms

    async def _subscribe_all(self, ws) -> None:
        for coin in self.symbols:
            for sub_type in ("l2Book", "trades", "activeAssetCtx"):
                await ws.send(
                    json.dumps(
                        {"method": "subscribe", "subscription": {"type": sub_type, "coin": coin}}
                    )
                )

    async def _ping_loop(self, ws, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.ping_interval_s)
                return  # stop was set
            except TimeoutError:
                pass
            try:
                await ws.send(json.dumps({"method": "ping"}))
            except Exception:  # noqa: BLE001 - connection likely dead, let the recv loop fail
                return
