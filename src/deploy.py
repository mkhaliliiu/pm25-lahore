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


def aqi_category(pm25):
    if pm25 is None or np.isnan(pm25):
        return "No data"
    return next(name for upper, name in AQI_BANDS if pm25 <= upper)


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
                x = _matrix(X, m["features"], m["median"])
                pm = ((x - m["mean"]) / m["scale"]) @ np.asarray(m["coef"]) + m["intercept"]
            else:
                pm = m.predict(_matrix(X, spec["features"]))          # LightGBM handles missing values
            out[f"pm25_h{h}"] = np.maximum(pm, 0)
            e = self.meta["event"][str(h)]
            prob = self.event[h].predict(_matrix(X, e["features"], e["median"]))
            out[f"event_prob_h{h}"] = prob
            out[f"event_warning_h{h}"] = (prob >= e["threshold"]).astype(int)
        return out
