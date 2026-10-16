"""Configuration loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import ROOT


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml from the repository root (or a given path)."""
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
