"""Live inputs from Open-Meteo (no API key needed): current weather and the CAMS PM2.5 forecast
for Lahore, turned into exactly the features the live model was trained on."""
import numpy as np
import pandas as pd
import requests

from src.config import LAT, LON, TZ
from src.features import add_features, add_live_features, add_targets

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
                "surface_pressure", "wind_speed_10m", "wind_direction_10m", "boundary_layer_height"]


def _get(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo returned no hourly data: {data}")
    frame = pd.DataFrame(data["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.set_index("time")


def fetch_live(now=None):
    """Returns (one-row feature frame for the latest full hour, hourly frame with the CAMS forecast,
    list of weather variables Open-Meteo did not provide)."""
    weather = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON, "timezone": TZ, "past_days": 2, "forecast_days": 4,
        "hourly": ",".join(WEATHER_VARS)})
    cams = _get("https://air-quality-api.open-meteo.com/v1/air-quality", {
        "latitude": LAT, "longitude": LON, "timezone": TZ, "past_days": 2, "forecast_days": 5,
        "hourly": "pm2_5", "domains": "cams_global"}).rename(columns={"pm2_5": "cams_pm25"})

    df = cams.join(weather.reindex(columns=WEATHER_VARS), how="left")
    missing = [v for v in WEATHER_VARS if df[v].isna().all()]
    df["pm25_obs"] = np.nan          # no live measurements: the live model does not use them
    df["pm25"] = np.nan
    df = add_live_features(add_targets(add_features(df)))

    now = pd.Timestamp.now(tz=TZ).tz_localize(None) if now is None else pd.Timestamp(now)
    issue = min(now.floor("h"), df.index.max() - pd.Timedelta(hours=72))
    return df.loc[[issue]], df[["cams_pm25"]], missing
