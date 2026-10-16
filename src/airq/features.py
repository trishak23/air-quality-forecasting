"""Feature engineering. All features at row t use information up to time t only."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER = ["temp_c", "wind_ms", "humidity", "pressure_hpa", "blh_m"]

AQI_BREAKS = [12.0, 35.4, 55.4, 150.4, 250.4]
AQI_LABELS = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]


def aqi_category(pm: np.ndarray | pd.Series) -> np.ndarray:
    """Map PM2.5 (ug/m3) to US-EPA-style AQI category index 0..5."""
    return np.searchsorted(AQI_BREAKS, np.asarray(pm, dtype=float), side="left")


def make_features(df: pd.DataFrame, horizon: int = 24, pm_lags: list[int] | None = None,
                  roll_windows: list[int] | None = None) -> pd.DataFrame:
    """Feature matrix (rows with incomplete lookback are NaN). Row t uses data <= t only."""
    pm_lags = pm_lags or [1, 2, 3, 6, 12, 24, 48, 168]
    roll_windows = roll_windows or [6, 24, 72]
    pm = df["pm25"]
    X = pd.DataFrame(index=df.index)
    X["pm_now"] = pm
    for L in pm_lags:
        X[f"pm_lag{L}"] = pm.shift(L)
    for w in roll_windows:
        r = pm.rolling(w, min_periods=w)
        X[f"pm_mean{w}"] = r.mean()
        X[f"pm_std{w}"] = r.std()
    X["pm_max24"] = pm.rolling(24, min_periods=24).max()
    X["pm_diff3"] = pm - pm.shift(3)
    for c in WEATHER:
        X[c] = df[c]
    X["wind_mean24"] = df["wind_ms"].rolling(24, min_periods=24).mean()
    X["pressure_d6"] = df["pressure_hpa"] - df["pressure_hpa"].shift(6)
    X["temp_d24"] = df["temp_c"] - df["temp_c"].shift(24)
    # calendar of the TARGET hour (known in advance)
    tgt = df.index + pd.Timedelta(hours=horizon)
    X["tgt_hour_sin"] = np.sin(2 * np.pi * tgt.hour / 24)
    X["tgt_hour_cos"] = np.cos(2 * np.pi * tgt.hour / 24)
    X["tgt_weekend"] = (tgt.dayofweek >= 5).astype(int)
    return X


def build_features(df: pd.DataFrame, horizon: int = 24, pm_lags: list[int] | None = None,
                   roll_windows: list[int] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is PM2.5 at t+horizon. Rows with NaN features/target are dropped."""
    X = make_features(df, horizon, pm_lags, roll_windows)
    y = df["pm25"].shift(-horizon).rename("y")
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def time_split(X: pd.DataFrame, y: pd.Series, test_fraction: float = 0.3,
               horizon: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split. The last `horizon` training rows are purged because
    their targets fall inside the test period."""
    cut = int(len(X) * (1 - test_fraction))
    cut_time = X.index[cut]
    tr = X.index < (cut_time - pd.Timedelta(hours=horizon))
    te = X.index >= cut_time
    return X[tr], X[te], y[tr], y[te]
