"""Load the saved model and forecast PM2.5 24h ahead from a history frame."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import ROOT
from .config import load_config
from .data import clean, load_data
from .features import AQI_LABELS, aqi_category, make_features


def forecast_from_history(history: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Forecast PM2.5 at t+horizon for each hour t of `history` with a complete lookback.

    The most recent hour yields a genuine forward forecast (its target is not needed).
    """
    X = make_features(history, bundle["horizon"]).dropna()
    pred = bundle["model"].predict(X[bundle["features"]])
    out = pd.DataFrame({"forecast_pm25": pred}, index=X.index + pd.Timedelta(hours=bundle["horizon"]))
    out["aqi_category"] = [AQI_LABELS[i] for i in aqi_category(pred)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="24h-ahead PM2.5 forecast from a history CSV")
    ap.add_argument("--csv", default=None, help="hourly CSV with the training columns (default: bundled sample)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--last", type=int, default=6, help="show the last N forecasts")
    a = ap.parse_args()
    cfg = load_config()
    bundle = joblib.load(Path(a.model) if a.model else ROOT / cfg["paths"]["model"])
    hist = clean(load_data(a.csv))
    print(forecast_from_history(hist, bundle).tail(a.last).round(1))


if __name__ == "__main__":
    main()
