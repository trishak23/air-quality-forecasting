"""Seeded synthetic hourly PM2.5 + weather generator.

The data are SYNTHETIC: PM2.5 is built as a log-scale process with seasonal
(winter heavy), diurnal (traffic peaks, afternoon mixing), weather (wind
dispersion, humidity, stagnation) and multi-day pollution-episode components,
plus AR(1) noise and a few sensor gaps.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT
from .config import load_config

log = logging.getLogger(__name__)


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Zero-mean AR(1) series with stationary std ``sigma``."""
    eps = rng.normal(0.0, sigma * np.sqrt(1 - phi**2), n)
    x = np.zeros(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def generate_air_quality(n_hours: int = 17520, start: str = "2022-01-01",
                         seed: int = 42, gap_rate: float = 0.004) -> pd.DataFrame:
    """Generate an hourly air-quality frame indexed by timestamp.

    Columns: pm25, temp_c, wind_ms, humidity, pressure_hpa, blh_m (boundary
    layer height proxy), episode (0/1 ground-truth flag, not used as a feature).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_hours, freq="h")
    hod = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    dow = idx.dayofweek.to_numpy()
    season = np.cos(2 * np.pi * (doy - 15) / 365.25)  # +1 mid-winter

    # --- weather drivers
    temp = 12 - 11 * season + 4.5 * np.sin(2 * np.pi * (hod - 9) / 24) + _ar1(rng, n_hours, 0.985, 3.0)
    wind = np.exp(0.9 + 0.35 * _ar1(rng, n_hours, 0.96, 1.0)) * (1 + 0.15 * np.sin(2 * np.pi * (hod - 8) / 24))
    wind = np.clip(wind, 0.2, 18)
    humidity = np.clip(70 - 1.0 * (temp - 12) - 12 * np.sin(2 * np.pi * (hod - 9) / 24)
                       + _ar1(rng, n_hours, 0.97, 9), 15, 100)
    pressure = 1013 + _ar1(rng, n_hours, 0.995, 7.0)
    blh = np.clip(500 + 700 * np.clip(np.sin(2 * np.pi * (hod - 6) / 24), 0, None)
                  * (0.6 - 0.4 * season) * np.exp(0.2 * _ar1(rng, n_hours, 0.95, 1.0)), 80, None)

    # --- pollution episodes: stagnation (high pressure + low wind) events lasting 1-5 days
    episode = np.zeros(n_hours)
    n_ep = int(n_hours / 24 / 18)
    starts = rng.choice(n_hours - 150, n_ep, replace=False)
    for s in starts:
        # episodes are more likely/stronger in cold months
        if rng.random() > 0.45 + 0.4 * season[s]:
            continue
        dur = int(rng.integers(24, 120))
        amp = rng.uniform(0.5, 1.3)
        ramp = np.sin(np.linspace(0, np.pi, dur)) ** 0.7
        episode[s:s + dur] = np.maximum(episode[s:s + dur], amp * ramp)
        wind[s:s + dur] *= 0.6          # stagnation reduces wind
        pressure[s:s + dur] += 6 * ramp  # and raises pressure
        blh[s:s + dur] *= 0.7

    # --- PM2.5 on log scale
    diurnal = 0.18 * np.exp(-0.5 * ((hod - 8) / 1.8) ** 2) + 0.25 * np.exp(-0.5 * ((hod - 20) / 2.5) ** 2) \
        - 0.12 * np.exp(-0.5 * ((hod - 15) / 3.0) ** 2)
    weekend = np.where(dow >= 5, -0.08, 0.0)
    log_pm = (2.55 + 0.38 * season + diurnal + weekend
              - 0.30 * np.log(wind) - 0.25 * np.log(blh / 600) + 0.006 * (humidity - 65)
              + 0.012 * (pressure - 1013) + 0.9 * episode
              + _ar1(rng, n_hours, 0.93, 0.22))
    pm = np.exp(log_pm)
    pm = pm * np.exp(rng.normal(0, 0.06, n_hours))  # measurement noise

    df = pd.DataFrame({"pm25": pm, "temp_c": temp, "wind_ms": wind, "humidity": humidity,
                       "pressure_hpa": pressure, "blh_m": blh, "episode": (episode > 0.05).astype(int)},
                      index=idx)
    df.index.name = "timestamp"
    # sensor gaps (PM2.5 only), 1-6 hours long
    n_gaps = int(gap_rate * n_hours)
    for s in rng.choice(n_hours - 8, n_gaps, replace=False):
        df.iloc[s:s + int(rng.integers(1, 7)), df.columns.get_loc("pm25")] = np.nan
    return df.round(3)


def generate_and_save(path: str | Path | None = None) -> Path:
    """Generate with config settings and write the sample CSV."""
    cfg = load_config()
    df = generate_air_quality(cfg["data"]["n_hours"], cfg["data"]["start"], cfg["seed"])
    out = Path(path) if path else ROOT / cfg["data"]["sample_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    log.info("wrote %s (%d rows)", out, len(df))
    return out


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the hourly CSV (generating it first if missing)."""
    cfg = load_config()
    p = Path(path) if path else ROOT / cfg["data"]["sample_path"]
    if not p.exists():
        generate_and_save(p)
    return pd.read_csv(p, index_col="timestamp", parse_dates=True)


def clean(df: pd.DataFrame, max_gap: int = 6) -> pd.DataFrame:
    """Fill short PM2.5 gaps by time interpolation (causal ffill; no look-ahead)."""
    out = df.copy()
    out["pm25"] = out["pm25"].ffill(limit=max_gap)
    return out
