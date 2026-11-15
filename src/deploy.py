"""Model files for the dashboard and the API, stored without pickle.

Each deployed model set is described by a meta file in models/:
  meta.json       full model (uses measured PM2.5; runs on the stored data snapshot)
  meta_live.json  live model (weather + CAMS from Open-Meteo only; runs on today's forecast)
PM2.5 models are Ridge (JSON: medians, scaler, coefficients) or LightGBM (native text format);
smog-warning models are LightGBM text files with imputer medians and a cost-based threshold.
These files load the same way whatever scikit-learn version is installed, and loading them
cannot run arbitrary code.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
HORIZONS = (24, 48, 72)

# US EPA PM2.5 categories (2024 revision), µg/m³ upper bounds
AQI_BANDS = [(9.0, "Good"), (35.4, "Moderate"), (55.4, "Unhealthy for Sensitive Groups"),
             (125.4, "Unhealthy"), (225.4, "Very Unhealthy"), (float("inf"), "Hazardous")]


# US EPA PM2.5 -> AQI breakpoints (2024 revision): (C_low, C_high, AQI_low, AQI_high)
AQI_BREAKS = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
              (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]


def aqi_category(pm25):
    if pm25 is None or np.isnan(pm25):
        return "No data"
    return next(name for upper, name in AQI_BANDS if pm25 <= upper)


def us_aqi(pm25):
    """US AQI from a PM2.5 concentration (µg/m³), the number IQAir and most apps show."""
    if pm25 is None or np.isnan(pm25):
        return np.nan
    c = np.floor(max(pm25, 0) * 10) / 10
    for lo, hi, a_lo, a_hi in AQI_BREAKS:
        if c <= hi:
            return int(round((a_hi - a_lo) / (hi - lo) * (c - lo) + a_lo))
    return 500


def ridge_predict(m, X):
    x = _matrix(X, m["features"], m["median"])
    return np.maximum(((x - m["mean"]) / m["scale"]) @ np.asarray(m["coef"]) + m["intercept"], 0)


class HourlyRidge:
    """Ridge models for every horizon 1..72 h (models/hourly_full.json or hourly_live.json),
    used for the hour-by-hour forecast curve on the live tab."""

    def __init__(self, kind="full", model_dir=MODEL_DIR):
        self.models = {int(h): m for h, m in json.loads((Path(model_dir) / f"hourly_{kind}.json").read_text()).items()}

    def predict(self, X):
        return pd.DataFrame({h: ridge_predict(m, X) for h, m in sorted(self.models.items())}, index=X.index)


def ridge_to_dict(pipe, features):
    imp, sc, rg = (pipe.named_steps[k] for k in ("simpleimputer", "standardscaler", "ridge"))
    return {"features": list(features), "median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(),
            "scale": sc.scale_.tolist(), "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_)}


def _matrix(X, features, medians=None):
    x = X.reindex(columns=features).astype(float).to_numpy()
    return x if medians is None else np.where(np.isnan(x), np.asarray(medians), x)


class Forecaster:
    """For each row of features (one issue time) returns PM2.5 at t+24/48/72 h, the probability
    that the 24-h mean there exceeds 125.5 µg/m³, and a 0/1 smog warning at the cost-based threshold."""

    def __init__(self, meta_name="meta.json", model_dir=MODEL_DIR):
        import lightgbm as lgb
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / meta_name).read_text())
        self.pm25, self.event = {}, {}
        for h in HORIZONS:
            spec = self.meta["pm25"][str(h)]
            if spec["format"] == "ridge_json":
                self.pm25[h] = json.loads((model_dir / spec["file"]).read_text())
            else:
                self.pm25[h] = lgb.Booster(model_file=str(model_dir / spec["file"]))
            self.event[h] = lgb.Booster(model_file=str(model_dir / self.meta["event"][str(h)]["file"]))

    @property
    def features(self):
        cols = []
        for h in HORIZONS:
            cols += self.meta["pm25"][str(h)]["features"] + self.meta["event"][str(h)]["features"]
        return list(dict.fromkeys(cols))          # unique, original order

    def predict(self, X):
        out = pd.DataFrame(index=X.index)
        for h in HORIZONS:
            spec, m = self.meta["pm25"][str(h)], self.pm25[h]
            if spec["format"] == "ridge_json":
                pm = ridge_predict(m, X)
            else:
                pm = m.predict(_matrix(X, spec["features"]))          # LightGBM handles missing values
            out[f"pm25_h{h}"] = np.maximum(pm, 0)
            e = self.meta["event"][str(h)]
            prob = self.event[h].predict(_matrix(X, e["features"], e["median"]))
            out[f"event_prob_h{h}"] = prob
            out[f"event_warning_h{h}"] = (prob >= e["threshold"]).astype(int)
        return out
