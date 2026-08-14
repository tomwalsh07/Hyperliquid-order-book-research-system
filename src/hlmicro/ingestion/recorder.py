"""Raw-first, append-only persistence for inbound WS messages.

Every message is written verbatim (plus a receive timestamp) before any
parsing happens, so a bug in the normalizer can never lose captured data —
worst case we re-run normalization against the raw log. Files are
hour-partitioned and opened in append mode, which makes stop/restart safe:
a restart just resumes appending to the current (or a fresh) hour file,
with no read-modify-write step that could corrupt or duplicate data.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import orjson

logger = logging.getLogger(__name__)


class RawRecorder:
    def __init__(
        self,
        raw_dir: Path | str,
        flush_every_n: int = 200,
        flush_every_s: float = 2.0,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.flush_every_n = flush_every_n
        self.flush_every_s = flush_every_s

        self._current_hour_key: tuple[str, str] | None = None
        self._fh = None
        self._since_flush = 0
        self._last_flush_mono = time.monotonic()
        self._lock = threading.Lock()

        self.messages_written = 0

    @staticmethod
    def _hour_key(recv_ts_ns: int) -> tuple[str, str]:
        dt = datetime.fromtimestamp(recv_ts_ns / 1e9, tz=UTC)
        return dt.strftime("%Y%m%d"), dt.strftime("%H")

    def _path_for(self, date_str: str, hour_str: str) -> Path:
        d = self.raw_dir / date_str
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{hour_str}.jsonl"

    def _rotate(self, date_str: str, hour_str: str) -> None:
        self._flush(force_fsync=True)
        if self._fh is not None:
            self._fh.close()
        path = self._path_for(date_str, hour_str)
        self._fh = open(path, "ab")  # append: safe across restarts, never truncates
        self._current_hour_key = (date_str, hour_str)
        logger.info("Recording to %s", path)

    def write(self, msg: dict, recv_ts_ns: int) -> None:
        date_str, hour_str = self._hour_key(recv_ts_ns)
        key = (date_str, hour_str)
        with self._lock:
            if key != self._current_hour_key:
                self._rotate(date_str, hour_str)
            line = orjson.dumps({"recv_ts_ns": recv_ts_ns, **msg})
            self._fh.write(line)
            self._fh.write(b"\n")
            self.messages_written += 1
            self._since_flush += 1

            now = time.monotonic()
            if (
                self._since_flush >= self.flush_every_n
                or (now - self._last_flush_mono) >= self.flush_every_s
            ):
                self._flush()

    def _flush(self, force_fsync: bool = False) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        if force_fsync or self._since_flush > 0:
            os.fsync(self._fh.fileno())
        self._since_flush = 0
        self._last_flush_mono = time.monotonic()

    def close(self) -> None:
        with self._lock:
            self._flush(force_fsync=True)
            if self._fh is not None:
                self._fh.close()
                self._fh = None
