"""Model training with time-series cross-validation."""
from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


class LogTargetGBM:
    """HistGradientBoosting on the log-scale residual log1p(y) - log1p(pm_now).

    The model therefore starts from the persistence forecast and learns a correction;
    predictions are returned in ug/m3. Requires a ``pm_now`` feature column."""

    def __init__(self, seed: int = 42, **params: Any) -> None:
        self.params = params
        self.model = HistGradientBoostingRegressor(random_state=seed, **params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogTargetGBM":
        self.model.fit(X, np.log1p(y) - np.log1p(X["pm_now"]).to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.model.predict(X) + np.log1p(X["pm_now"].to_numpy()))


class LogTargetRidge:
    """Standardised ridge on log1p(PM2.5)."""

    def __init__(self, alpha: float = 5.0) -> None:
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogTargetRidge":
        self.model.fit(X, np.log1p(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.expm1(self.model.predict(X))


def cv_select(X: pd.DataFrame, y: pd.Series, grid: list[dict], n_splits: int, seed: int,
              gap: int = 24) -> tuple[dict, list[dict]]:
    """Expanding-window CV over the training period; returns best params and CV table."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows = []
    for params in grid:
        maes = []
        for tr, va in tscv.split(X):
            m = LogTargetGBM(seed, **params).fit(X.iloc[tr], y.iloc[tr])
            maes.append(mean_absolute_error(y.iloc[va], m.predict(X.iloc[va])))
        rows.append({"params": params, "cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes))})
        log.info("CV %s -> MAE %.3f", params, np.mean(maes))
    best = min(rows, key=lambda r: r["cv_mae_mean"])
    return best["params"], rows


def save_model(model: Any, feature_names: list[str], path, horizon: int, meta: dict | None = None) -> None:
    """Persist model with its feature list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_names, "horizon": horizon, "meta": meta or {}}, path)
