#!/usr/bin/env python
"""Unattended live market-data collector.

    python scripts/collect.py --symbols BTC,ETH,SOL --out data/raw/

Safe to stop (Ctrl+C, SIGTERM, or a hard process kill) and restart at any
time: raw capture files are append-only and hour-partitioned, so a restart
never corrupts or duplicates prior data (see RawRecorder docstring).

Public market data only. No API keys required.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hlmicro.config import load_config  # noqa: E402
from hlmicro.ingestion.recorder import RawRecorder  # noqa: E402
from hlmicro.ingestion.rest_poller import poll_predicted_fundings  # noqa: E402
from hlmicro.ingestion.ws_client import HyperliquidWSClient  # noqa: E402


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


async def status_logger(
    recorder: RawRecorder, client: HyperliquidWSClient, stop_event: asyncio.Event, interval_s: float
) -> None:
    logger = logging.getLogger("collect.status")
    start = time.monotonic()
    last_count = 0
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            pass
        elapsed = time.monotonic() - start
        n = recorder.messages_written
        rate = (n - last_count) / interval_s
        last_count = n
        logger.info(
            "status: elapsed=%.0fs written=%d rate=%.1f/s reconnects=%d gaps=%d",
            elapsed,
            n,
            rate,
            client.stats.reconnects,
            client.stats.gaps_detected,
        )


async def run(symbols: list[str], out_dir: Path, cfg: dict, duration_hours: float | None) -> None:
    ing_cfg = cfg["ingestion"]
    recorder = RawRecorder(out_dir)
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # Windows: no add_signal_handler; KeyboardInterrupt is caught in __main__ instead

    async def on_message(msg: dict, recv_ts_ns: int) -> None:
        recorder.write(msg, recv_ts_ns)

    client = HyperliquidWSClient(
        symbols=symbols,
        on_message=on_message,
        ws_url=ing_cfg["ws_url"],
        ping_interval_s=ing_cfg["ping_interval_s"],
        reconnect_backoff_base_s=ing_cfg["reconnect_backoff_base_s"],
        reconnect_backoff_max_s=ing_cfg["reconnect_backoff_max_s"],
        reconnect_jitter_s=ing_cfg["reconnect_jitter_s"],
        gap_warn_threshold_s=ing_cfg["gap_warn_threshold_s"],
    )

    tasks = [
        asyncio.create_task(client.run(stop_event)),
        asyncio.create_task(
            poll_predicted_fundings(symbols, recorder, stop_event, interval_s=60.0)
        ),
        asyncio.create_task(status_logger(recorder, client, stop_event, interval_s=60.0)),
    ]

    if duration_hours is not None:

        async def _timer():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=duration_hours * 3600)
            except TimeoutError:
                logging.getLogger("collect").info("Duration limit reached, stopping.")
                stop_event.set()

        tasks.append(asyncio.create_task(_timer()))

    try:
        await asyncio.gather(*tasks)
    finally:
        stop_event.set()
        recorder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="BTC,ETH,SOL", help="Comma-separated symbol list")
    parser.add_argument(
        "--out", default=None, help="Raw output dir (default: config ingestion.raw_dir)"
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--duration-hours", type=float, default=None, help="Auto-stop after N hours"
    )
    parser.add_argument("--log-file", default="logs/collect.log")
    args = parser.parse_args()

    setup_logging(Path(args.log_file))
    cfg = load_config(args.config)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.out) if args.out else Path(cfg["ingestion"]["raw_dir"])

    logging.getLogger("collect").info(
        "Starting collector: symbols=%s out=%s duration_hours=%s",
        symbols,
        out_dir,
        args.duration_hours,
    )

    try:
        asyncio.run(run(symbols, out_dir, cfg, args.duration_hours))
    except KeyboardInterrupt:
        logging.getLogger("collect").info("Interrupted, shutting down.")


if __name__ == "__main__":
    main()
