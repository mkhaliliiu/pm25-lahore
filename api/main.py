"""FastAPI prediction endpoint for the Lahore PM2.5 forecaster.

Run from the repository root:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive documentation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deploy import HORIZONS, Forecaster, aqi_category  # noqa: E402

app = FastAPI(
    title="Lahore PM2.5 forecast API",
    version="1.0.0",
    description="PM2.5 forecasts 24/48/72 h ahead (Ridge + CAMS) and severe-smog warnings "
                "(24-h mean > 125.5 µg/m³, LightGBM + SMOTE) for Lahore. /live forecasts from today's "
                "Open-Meteo weather and CAMS forecast; /forecast and /predict use the full model.",
)
MODEL = Forecaster("meta.json")
LIVE_MODEL = Forecaster("meta_live.json")
RECENT = pd.read_csv(ROOT / "app" / "data" / "recent_features.csv.gz", parse_dates=["time"], index_col="time")


class FeatureRow(BaseModel):
    issued_at: str | None = Field(None, description="Issue time (Pakistan time), optional, e.g. 2025-03-01T08:00")
    features: dict[str, float | None] = Field(
        ..., description="Model inputs at the issue time. Missing inputs are filled with training medians. "
                         "GET /features/example returns a complete example.")


def _response(X, issued_at, model=MODEL):
    p = model.predict(X).iloc[0]
    out = []
    for h in HORIZONS:
        target = (pd.Timestamp(issued_at) + pd.Timedelta(hours=h)).isoformat() if issued_at else None
        pm = float(p[f"pm25_h{h}"])
        out.append({"horizon_h": h, "target_time": target, "pm25_ugm3": round(pm, 1),
                    "aqi_category": aqi_category(pm),
                    "smog_probability": round(float(p[f"event_prob_h{h}"]), 3),
                    "smog_warning": bool(p[f"event_warning_h{h}"])})
    return {"issued_at": issued_at, "forecasts": out,
            "models": {"name": model.meta["name"], "pm25": model.meta["regression_model"],
                       "warning": model.meta["event_model"], "warning_definition": model.meta["event_definition"]}}


@app.get("/health")
def health():
    return {"status": "ok", "trained_on": MODEL.meta["trained_on"],
            "snapshot": [str(RECENT.index.min()), str(RECENT.index.max())]}


@app.get("/live")
def live():
    """Live forecast for Lahore from today's Open-Meteo weather and CAMS forecast (weather + CAMS model)."""
    from src.live import fetch_live
    try:
        row, _, missing = fetch_live()
    except Exception as e:
        raise HTTPException(503, f"Open-Meteo unavailable: {e}")
    out = _response(row[LIVE_MODEL.features], row.index[0].isoformat(), LIVE_MODEL)
    out["inputs_not_provided_by_open_meteo"] = missing
    return out


@app.get("/features/example")
def feature_example():
    row = RECENT.iloc[-1][MODEL.features]
    return {"issued_at": RECENT.index[-1].isoformat(),
            "features": {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}}


@app.get("/forecast")
def forecast(time: str | None = Query(None, description="Issue time inside the data snapshot; default = latest")):
    """Forecast from the stored data snapshot (the station's feed ended in March 2025)."""
    t = RECENT.index[-1] if time is None else pd.Timestamp(time).floor("h")
    if t not in RECENT.index:
        raise HTTPException(404, f"No data for {t}. Snapshot covers {RECENT.index.min()} to {RECENT.index.max()}.")
    return _response(RECENT.loc[[t], MODEL.features], t.isoformat())


@app.post("/predict")
def predict(row: FeatureRow):
    """Forecast from features you send (for example from your own data pipeline)."""
    unknown = sorted(set(row.features) - set(MODEL.features))
    if unknown:
        raise HTTPException(422, f"Unknown feature names: {unknown}")
    X = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in row.features.items()}])
    return _response(X, row.issued_at)
