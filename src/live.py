"""Live inputs for Lahore.

* Open-Meteo (no key): weather and the CAMS PM2.5 forecast, past days and next days.
* OpenAQ (key): hourly PM2.5 from the Lahore low-cost sensor network (AirGradient, Hawanama).
  The hourly median across sensors is calibrated to the US Embassy reference monitor
  (models/meta_hourly.json, fitted in notebook 08) and used like the training measurements.

If measurements are available, forecasts come from the full model (measured PM2.5 + weather + CAMS).
If not, they fall back to the live model (weather + CAMS only).
"""
import json
import time

import numpy as np
import pandas as pd
import requests

from src.config import LAT, LON, ROOT, TZ
from src.features import add_cams_horizons, add_features, add_live_features, add_targets

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
                "surface_pressure", "wind_speed_10m", "wind_direction_10m", "boundary_layer_height"]
OPENAQ = "https://api.openaq.org/v3"
HOURLY = range(1, 73)


# ----------------------------------------------------------------------------- Open-Meteo
def _open_meteo(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo returned no hourly data: {data}")
    frame = pd.DataFrame(data["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.set_index("time")


def fetch_open_meteo(past_days=10):
    weather = _open_meteo("https://api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON, "timezone": TZ, "past_days": past_days, "forecast_days": 4,
        "hourly": ",".join(WEATHER_VARS)})
    cams = _open_meteo("https://air-quality-api.open-meteo.com/v1/air-quality", {
        "latitude": LAT, "longitude": LON, "timezone": TZ, "past_days": past_days, "forecast_days": 5,
        "hourly": "pm2_5", "domains": "cams_global"}).rename(columns={"pm2_5": "cams_pm25"})
    df = cams.join(weather.reindex(columns=WEATHER_VARS), how="left")
    return df, [v for v in WEATHER_VARS if df[v].isna().all()]


# ----------------------------------------------------------------------------- OpenAQ
def openaq_hours(api_key, sensor_id, start, end, tries=3):
    """Hourly values of one OpenAQ sensor between two UTC times -> Series indexed by Pakistan time."""
    s = requests.Session()
    s.headers["X-API-Key"] = api_key
    rows, page = [], 1
    while True:
        for k in range(tries):
            try:
                r = s.get(f"{OPENAQ}/sensors/{sensor_id}/hours", timeout=60, params={
                    "datetime_from": pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "datetime_to": pd.Timestamp(end).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 1000, "page": page})
                if r.status_code == 429:
                    time.sleep(20)
                    continue
                r.raise_for_status()
                res = r.json().get("results", [])
                break
            except requests.RequestException:
                time.sleep(3 * (k + 1))
        else:
            return None                                       # this sensor failed; skip it
        rows += [(x["period"]["datetimeFrom"]["utc"], x["value"]) for x in res]
        if len(res) < 1000:
            break
        page += 1
    if not rows:
        return pd.Series(dtype=float)
    t = pd.to_datetime([a for a, _ in rows], utc=True).tz_convert(TZ).tz_localize(None).floor("h")
    return pd.Series([v for _, v in rows], index=t, dtype=float).groupby(level=0).median()


def network_median(wide, min_sensors):
    """Hourly median across sensors, only where enough sensors reported; implausible values dropped."""
    wide = wide.where((wide > 0) & (wide < 1500))
    n = wide.notna().sum(axis=1)
    return wide.median(axis=1).where(n >= min_sensors), n


def fetch_network(api_key, sensor_ids, past_days=10, min_sensors=4):
    end = pd.Timestamp.now(tz="UTC").ceil("h")
    start = end - pd.Timedelta(days=past_days)
    series = {}
    for sid in sensor_ids:
        x = openaq_hours(api_key, sid, start, end)
        if x is not None and len(x):
            series[sid] = x
    if not series:
        return None, None, 0
    med, n = network_median(pd.DataFrame(series), min_sensors)
    return med, n, len(series)


# ----------------------------------------------------------------------------- live frame
def load_live_config():
    meta = json.loads((ROOT / "models" / "meta_hourly.json").read_text())
    sensors = pd.read_csv(ROOT / "app" / "data" / "live_sensors.csv")
    return meta, sensors


def build_live(api_key=None, now=None, past_days=10):
    """Everything the live tab and the /live endpoint need, in one dict."""
    meta, sensors = load_live_config()
    df, missing = fetch_open_meteo(past_days)
    cal = meta["calibration"]
    measured_raw, n_sensors, n_responding = None, None, 0
    if api_key:
        measured_raw, n_sensors, n_responding = fetch_network(api_key, sensors["sensor_id"].tolist(),
                                                              past_days, meta["min_sensors"])
    df["pm25_raw_network"] = measured_raw.reindex(df.index) if measured_raw is not None else np.nan
    df["n_sensors"] = n_sensors.reindex(df.index) if n_sensors is not None else 0
    df["pm25_obs"] = cal["intercept"] + cal["slope"] * df["pm25_raw_network"]
    df["pm25_obs"] = df["pm25_obs"].clip(lower=1)
    df["pm25"] = df["pm25_obs"].interpolate(limit=3, limit_area="inside")
    df = add_cams_horizons(add_live_features(add_targets(add_features(df))), HOURLY)

    now = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h") if now is None else pd.Timestamp(now)
    last_ok = df.index[df["pm25_obs"].notna() & (df.index <= now)]
    anchored = len(last_ok) > 0 and (now - last_ok.max()) <= pd.Timedelta(hours=4)
    issue = last_ok.max() if anchored else now
    issue = min(issue, df.index.max() - pd.Timedelta(hours=72))
    row = df.loc[[issue]]
    anchored = anchored and row[["pm25", "pm25_lag24", "pm25_mean24"]].notna().all(axis=1).iloc[0]
    return {"frame": df, "row": row, "issue": issue, "anchored": bool(anchored), "missing_weather": missing,
            "sensors_responding": n_responding, "sensors_total": len(sensors), "calibration": cal}


def verification(df, hourly_full, now, days=7, h=24):
    """Hindcast check for the last `days` days: the h-hour-ahead forecast made from live inputs at
    each past hour, against the calibrated network measurement at the target hour."""
    issues = df.index[(df.index >= now - pd.Timedelta(days=days) - pd.Timedelta(hours=h))
                      & (df.index <= now - pd.Timedelta(hours=h))]
    X = df.loc[issues]
    ok = X[["pm25", "pm25_lag24", "pm25_mean24"]].notna().all(axis=1)
    X = X[ok]
    if X.empty:
        return pd.DataFrame()
    pred = hourly_full.models[h]
    from src.deploy import ridge_predict
    out = pd.DataFrame({"target_time": X.index + pd.Timedelta(hours=h),
                        "forecast": ridge_predict(pred, X),
                        "raw_cams": X[f"cams_h{h}"].to_numpy()})
    out["measured"] = df["pm25_obs"].reindex(out["target_time"]).to_numpy()
    return out.dropna(subset=["measured"])
