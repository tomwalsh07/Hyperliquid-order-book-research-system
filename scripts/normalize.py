#!/usr/bin/env python
"""Normalize raw captured WS messages into typed Parquet.

    python scripts/normalize.py --raw-dir data/raw --out data/processed

Safe to re-run at any time, including while the live collector is still
writing to data/raw/ — see hlmicro.ingestion.normalize for the idempotency
contract.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hlmicro.ingestion.normalize import normalize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    rows = normalize(Path(args.raw_dir), Path(args.out))
    total = sum(rows.values())
    logging.info("Done. Rows written this run: %s (total=%d)", rows, total)


if __name__ == "__main__":
    main()
