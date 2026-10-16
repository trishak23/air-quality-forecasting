"""Metrics, error analysis and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from .features import AQI_LABELS, aqi_category  # noqa: E402


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE / RMSE / R2 in ug/m3."""
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def category_metrics(y_true, y_pred) -> dict[str, float]:
    """AQI-category accuracy, within-one-category accuracy, and F1 for 'USG or worse' alerts."""
    ct, cp = aqi_category(y_true), aqi_category(y_pred)
    alert_t, alert_p = ct >= 2, cp >= 2
    return {"category_accuracy": float((ct == cp).mean()),
            "within_one_category": float((np.abs(ct - cp) <= 1).mean()),
            "alert_f1_usg_plus": float(f1_score(alert_t, alert_p, zero_division=0)),
            "alert_rate_true": float(alert_t.mean())}


def full_report(y_true, y_pred) -> dict[str, float]:
    return {**regression_metrics(y_true, y_pred), **category_metrics(y_true, y_pred)}


def plot_forecast_window(index, y_true, preds: dict[str, np.ndarray], path: Path, days: int = 14) -> None:
    """Plot the first `days` of the test period: actual vs forecasts (indexed by target time)."""
    n = days * 24
    fig, ax = plt.subplots(figsize=(12, 4))
    t = index[:n] + pd.Timedelta(hours=24)
    ax.plot(t, np.asarray(y_true)[:n], color="black", lw=1.4, label="Actual")
    for name, p in preds.items():
        ax.plot(t, np.asarray(p)[:n], lw=1, alpha=0.85, label=name)
    ax.set_ylabel("PM2.5 (ug/m3)"); ax.set_title("24h-ahead forecast, first 14 days of the test period")
    ax.legend(ncol=3); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_scatter(y_true, preds: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.5), sharex=True, sharey=True)
    lim = float(np.percentile(y_true, 99.8))
    for ax, (name, p) in zip(np.atleast_1d(axes), preds.items()):
        ax.scatter(y_true, p, s=3, alpha=0.25)
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(name); ax.set_xlabel("Actual PM2.5")
    np.atleast_1d(axes)[0].set_ylabel("Predicted PM2.5")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(y_true, y_pred, path: Path, title: str) -> None:
    ct, cp = aqi_category(y_true), aqi_category(y_pred)
    labels = list(range(len(AQI_LABELS)))
    cm = confusion_matrix(ct, cp, labels=labels)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(labels); ax.set_yticks(labels)
    ax.set_xticklabels(AQI_LABELS, rotation=45, ha="right"); ax.set_yticklabels(AQI_LABELS)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("Actual category"); ax.set_title(title)
    fig.colorbar(im, label="row-normalised share"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_importance(names: list[str], means: np.ndarray, stds: np.ndarray, path: Path, top: int = 15) -> None:
    order = np.argsort(means)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[order], means[order], xerr=stds[order], color="#3b7dd8")
    ax.set_xlabel("Permutation importance (increase in MAE, ug/m3)")
    ax.set_title("Top features (test set)"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def error_by_hour(index, y_true, preds: dict[str, np.ndarray], path: Path) -> dict[str, list[float]]:
    """MAE by target hour-of-day for each model; also plotted."""
    tgt_hour = (index + pd.Timedelta(hours=24)).hour
    out = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, p in preds.items():
        s = pd.Series(np.abs(np.asarray(y_true) - np.asarray(p))).groupby(np.asarray(tgt_hour)).mean()
        out[name] = [float(v) for v in s.values]
        ax.plot(s.index, s.values, marker="o", ms=3, label=name)
    ax.set_xlabel("Target hour of day"); ax.set_ylabel("MAE (ug/m3)"); ax.legend()
    ax.set_title("Forecast error by target hour"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return out
