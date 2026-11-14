"""Cleaning, feature engineering and targets. Shared by the notebooks and the dashboard.

Leakage rule: every feature at time t uses only information available at time t.
The one exception is the CAMS value for the target hour, which is a forecast that CAMS
issues in advance (see the paper's limitations on archived CAMS lead time).
"""
import numpy as np
import pandas as pd

from src.config import CAMS_START, DATA_RAW, EVENT_THRESHOLD, HORIZONS, TZ

MAX_GAP_FILL = 3        # hours: gaps up to this length are linearly interpolated
STUCK_RUN = 8           # hours: identical consecutive readings this long are treated as a stuck sensor
LAGS = (1, 3, 6, 24, 48)
WINDOWS = (6, 24)


# ----------------------------------------------------------------------------- loading
def load_openaq_chunks(sensor_id=None):
    """Read the monthly OpenAQ files and report duplicates before collapsing them."""
    files = sorted((DATA_RAW / "openaq_chunks").glob(f"{sensor_id or '*'}_*.csv"))
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).dropna(subset=["time_utc"])
    raw["time"] = (pd.to_datetime(raw["time_utc"], utc=True, format="ISO8601")
                   .dt.tz_convert(TZ).dt.tz_localize(None))
    report = {
        "raw_rows": len(raw),
        "exact_duplicate_rows": int(raw.duplicated(["time_utc", "pm25"]).sum()),
        "rows_not_on_the_hour": int((raw["time"] != raw["time"].dt.floor("h")).sum()),
    }
    raw["time"] = raw["time"].dt.floor("h")
    report["unique_hours"] = int(raw["time"].nunique())
    report["hours_with_conflicting_values"] = int(
        (raw.groupby("time")["pm25"].nunique() > 1).sum())
    pm = raw.groupby("time")["pm25"].median()
    return pm, report


def load_weather_and_cams():
    era5 = pd.read_csv(DATA_RAW / "era5_weather.csv", parse_dates=["time"])
    cams = pd.read_csv(DATA_RAW / "cams_pm25.csv", parse_dates=["time"])
    return era5.set_index("time"), cams.set_index("time")[["cams_pm25"]]


# ----------------------------------------------------------------------------- cleaning
def clean_pm25(pm):
    """Remove impossible and stuck readings; keep genuine extremes (they are the events)."""
    pm = pm.copy()
    report = {"non_positive": int((pm <= 0).sum())}
    pm[pm <= 0] = np.nan

    run_id = (pm != pm.shift()).cumsum()
    run_len = pm.groupby(run_id).transform("size")
    stuck = (run_len >= STUCK_RUN) & pm.notna()
    report["stuck_hours"] = int(stuck.sum())
    pm[stuck] = np.nan
    return pm, report


def to_hourly_grid(pm, era5, cams):
    start, end = pm.index.min().floor("D"), pm.index.max().ceil("D") - pd.Timedelta(hours=1)
    grid = pd.date_range(start, end, freq="h", name="time")
    df = pd.DataFrame(index=grid)
    df["pm25_obs"] = pm.reindex(grid)                       # cleaned, never imputed: used for targets
    df["pm25"] = df["pm25_obs"].interpolate(limit=MAX_GAP_FILL, limit_area="inside")  # for features
    df = df.join(era5, how="left").join(cams, how="left")
    df.loc[df.index < pd.Timestamp(CAMS_START), "cams_pm25"] = np.nan
    return df


# ----------------------------------------------------------------------------- features
def add_features(df):
    out = df.copy()
    for lag in LAGS:
        out[f"pm25_lag{lag}"] = out["pm25"].shift(lag)
    for w in WINDOWS:
        roll = out["pm25"].rolling(w, min_periods=int(w * 0.75))
        out[f"pm25_mean{w}"] = roll.mean()
        out[f"pm25_std{w}"] = roll.std()
    out["pm25_diff24"] = out["pm25"] - out["pm25_lag24"]

    rad = np.deg2rad(out["wind_direction_10m"])
    out["wind_sin"], out["wind_cos"] = np.sin(rad), np.cos(rad)
    # Ventilation index: mixing height x wind speed (m2/s). Low values trap pollution (inversions).
    out["ventilation"] = out["boundary_layer_height"] * out["wind_speed_10m"] / 3.6
    out["log_ventilation"] = np.log1p(out["ventilation"])
    out["rain_24h"] = out["precipitation"].rolling(24, min_periods=1).sum()

    hour, month = out.index.hour, out.index.month
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    out["dayofweek"] = out.index.dayofweek
    out["smog_season"] = month.isin([10, 11, 12, 1, 2]).astype(int)
    return out


def add_targets(df):
    """Regression target: observed PM2.5 at t+h. Event target: observed 24-h mean at t+h > threshold.
    CAMS for the target hour (cams_h{h}) is the forecast CAMS issues for that hour."""
    out = df.copy()
    mean24_obs = out["pm25_obs"].rolling(24, min_periods=18).mean()
    cams_mean24 = out["cams_pm25"].rolling(24, min_periods=18).mean()
    out["mean24_obs"] = mean24_obs
    for h in HORIZONS:
        out[f"y_pm25_h{h}"] = out["pm25_obs"].shift(-h)
        event = (mean24_obs.shift(-h) > EVENT_THRESHOLD).astype(float)
        out[f"y_event_h{h}"] = event.where(mean24_obs.shift(-h).notna())
        out[f"cams_h{h}"] = out["cams_pm25"].shift(-h)
        out[f"cams_mean24_h{h}"] = cams_mean24.shift(-h)   # CAMS forecast of the target-day mean
    return out


FEATURES_BASE = (
    ["pm25"] + [f"pm25_lag{l}" for l in LAGS]
    + [f"pm25_mean{w}" for w in WINDOWS] + [f"pm25_std{w}" for w in WINDOWS] + ["pm25_diff24"]
    + ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
       "wind_speed_10m", "wind_sin", "wind_cos", "boundary_layer_height",
       "log_ventilation", "precipitation", "rain_24h"]
    + ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "smog_season"]
)


def feature_list(h, with_cams):
    return FEATURES_BASE + ([f"cams_h{h}", f"cams_mean24_h{h}"] if with_cams else [])


# ----------------------------------------------------------------------------- live model
# The US Embassy feed ended in March 2025, so the live model cannot use measured PM2.5.
# It uses only what Open-Meteo provides in real time: weather now, the CAMS value now,
# and the CAMS forecast for the target hour (the CAMS-correction part of the full model).
LIVE_BASE = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m",
    "wind_sin", "wind_cos", "boundary_layer_height", "log_ventilation", "precipitation", "rain_24h",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "smog_season",
    "cams_pm25", "cams_mean24_now",
]


def add_live_features(df):
    out = df.copy()
    out["cams_mean24_now"] = out["cams_pm25"].rolling(24, min_periods=18).mean()
    return out


def feature_list_live(h):
    return LIVE_BASE + [f"cams_h{h}", f"cams_mean24_h{h}"]
