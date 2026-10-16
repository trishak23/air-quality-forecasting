"""End-to-end pipeline: data -> features -> CV -> train -> evaluate -> reports."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from airq import evaluate as ev  # noqa: E402
from airq.config import load_config  # noqa: E402
from airq.data import clean, load_data  # noqa: E402
from airq.features import build_features, time_split  # noqa: E402
from airq.train import LogTargetGBM, LogTargetRidge, cv_select, save_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


def main() -> None:
    cfg = load_config()
    seed, H = cfg["seed"], cfg["horizon_hours"]
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    df = clean(raw)
    X, y = build_features(df, H, cfg["features"]["pm_lags"], cfg["features"]["roll_windows"])
    Xtr, Xte, ytr, yte = time_split(X, y, cfg["split"]["test_fraction"], H)
    log.info("train %d rows (%s -> %s) | test %d rows (%s -> %s)", len(Xtr), Xtr.index[0].date(),
             Xtr.index[-1].date(), len(Xte), Xte.index[0].date(), Xte.index[-1].date())

    best, cv_rows = cv_select(Xtr, ytr, cfg["model"]["param_grid"], cfg["model"]["cv_splits"], seed, gap=H)
    log.info("best params %s", best)
    gbm = LogTargetGBM(seed, **best).fit(Xtr, ytr)
    ridge = LogTargetRidge().fit(Xtr, ytr)

    preds = {"Persistence": Xte["pm_now"].to_numpy(),
             "Ridge (log)": ridge.predict(Xte),
             "GBM (log)": gbm.predict(Xte)}
    results = {k: ev.full_report(yte, v) for k, v in preds.items()}
    for k, v in results.items():
        log.info("%-12s %s", k, {m: round(x, 3) for m, x in v.items()})

    # Error analysis: episode vs non-episode hours (ground-truth flag at target time)
    ep = raw["episode"].reindex(yte.index + np.timedelta64(H, "h")).to_numpy().astype(bool)
    ep_report = {k: {"episode_mae": float(np.abs(yte.to_numpy() - v)[ep].mean()),
                     "non_episode_mae": float(np.abs(yte.to_numpy() - v)[~ep].mean())} for k, v in preds.items()}
    hi = yte.to_numpy() >= 35.5
    hi_report = {k: float(np.abs(yte.to_numpy() - v)[hi].mean()) for k, v in preds.items()}

    pi = permutation_importance(gbm.model, Xte.iloc[::3], np.log1p(yte.iloc[::3]),
                                scoring="neg_mean_absolute_error", n_repeats=5, random_state=seed)

    ev.plot_forecast_window(Xte.index, yte, {"Persistence": preds["Persistence"], "GBM": preds["GBM (log)"]},
                            fig_dir / "forecast_window.png")
    ev.plot_scatter(yte, preds, fig_dir / "scatter_pred_vs_actual.png")
    ev.plot_confusion(yte, preds["GBM (log)"], fig_dir / "confusion_gbm.png", "AQI category: GBM, 24h ahead")
    ev.plot_confusion(yte, preds["Persistence"], fig_dir / "confusion_persistence.png", "AQI category: persistence")
    ev.plot_importance(list(Xte.columns), pi.importances_mean, pi.importances_std, fig_dir / "feature_importance.png")
    hour_mae = ev.error_by_hour(Xte.index, yte, {"Persistence": preds["Persistence"], "GBM": preds["GBM (log)"]},
                                fig_dir / "mae_by_hour.png")

    save_model(gbm, list(Xtr.columns), ROOT / cfg["paths"]["model"], H, {"best_params": best})

    metrics = {
        "data": {"synthetic": True, "n_hours": int(len(raw)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                 "test_start": str(Xte.index[0]), "pm25_missing_share_raw": float(raw["pm25"].isna().mean())},
        "horizon_hours": H,
        "cv": {"best_params": best, "table": cv_rows},
        "test": results,
        "mae_by_regime": {"episode_hours": ep_report, "high_pm_hours_ge_35.5_mae": hi_report,
                          "episode_share_test": float(ep.mean())},
        "gbm_mae_improvement_over_persistence_pct":
            float(100 * (1 - results["GBM (log)"]["mae"] / results["Persistence"]["mae"])),
        "top_features": [[Xte.columns[i], float(pi.importances_mean[i])]
                         for i in np.argsort(pi.importances_mean)[::-1][:8]],
        "mae_by_target_hour": hour_mae,
    }
    out = ROOT / cfg["paths"]["metrics"]
    out.write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
