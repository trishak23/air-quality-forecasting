import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airq.data import clean, generate_air_quality  # noqa: E402
from airq.features import aqi_category, build_features, make_features, time_split  # noqa: E402
from airq.train import LogTargetGBM, save_model  # noqa: E402
from airq.predict import forecast_from_history  # noqa: E402
from airq.evaluate import category_metrics  # noqa: E402


class TestData(unittest.TestCase):
    def test_deterministic_and_shape(self):
        a = generate_air_quality(2000, seed=1)
        b = generate_air_quality(2000, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 2000)
        self.assertTrue((a["pm25"].dropna() > 0).all())

    def test_winter_worse_than_summer(self):
        d = generate_air_quality(17520, seed=3)
        m = d["pm25"].groupby(d.index.month).mean()
        self.assertGreater(m[[12, 1, 2]].mean(), m[[6, 7, 8]].mean())


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = clean(generate_air_quality(1500, seed=2))

    def test_aqi_categories(self):
        np.testing.assert_array_equal(aqi_category([5, 12.0, 12.1, 40, 100, 200, 300]), [0, 0, 1, 2, 3, 4, 5])

    def test_no_future_leakage(self):
        t = self.df.index[800]
        base = make_features(self.df).loc[t]
        mod = self.df.copy()
        mod.loc[mod.index > t, ["pm25", "wind_ms", "temp_c"]] *= 5
        pd.testing.assert_series_equal(base, make_features(mod).loc[t])

    def test_target_is_24h_ahead(self):
        X, y = build_features(self.df, 24)
        t = X.index[10]
        self.assertAlmostEqual(y.loc[t], self.df["pm25"].loc[t + pd.Timedelta(hours=24)])

    def test_time_split_ordering_and_purge(self):
        X, y = build_features(self.df, 24)
        Xtr, Xte, ytr, yte = time_split(X, y, 0.3, 24)
        self.assertLess(Xtr.index.max() + pd.Timedelta(hours=24), Xte.index.min())


class TestModel(unittest.TestCase):
    def test_gbm_beats_persistence_and_predict_roundtrip(self):
        df = clean(generate_air_quality(6000, seed=5))
        X, y = build_features(df, 24)
        Xtr, Xte, ytr, yte = time_split(X, y, 0.3, 24)
        m = LogTargetGBM(0, max_iter=80, learning_rate=0.1).fit(Xtr, ytr)
        mae_m = np.abs(m.predict(Xte) - yte).mean()
        mae_p = np.abs(Xte["pm_now"] - yte).mean()
        self.assertLess(mae_m, mae_p)
        with tempfile.TemporaryDirectory() as d:
            import joblib
            p = Path(d) / "m.joblib"
            save_model(m, list(Xtr.columns), p, 24)
            out = forecast_from_history(df, joblib.load(p))
        self.assertGreater(out.index.max(), df.index.max())  # genuine forward forecast
        self.assertTrue(out["forecast_pm25"].notna().all())

    def test_category_metrics_perfect(self):
        r = category_metrics(np.array([5, 40, 80]), np.array([5, 40, 80]))
        self.assertEqual(r["category_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
