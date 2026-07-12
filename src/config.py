"""Config loader (YAML)."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}
