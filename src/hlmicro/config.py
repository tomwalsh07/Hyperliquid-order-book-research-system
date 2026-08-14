"""Loads config/config.yaml. No secrets live here — see .env.example for the
(optional) environment overrides, none of which are required to run."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)

    ws_url = os.environ.get("HLMICRO_WS_URL")
    if ws_url:
        cfg["ingestion"]["ws_url"] = ws_url
    data_dir = os.environ.get("HLMICRO_DATA_DIR")
    if data_dir:
        cfg["ingestion"]["raw_dir"] = data_dir

    return cfg
