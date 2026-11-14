# %% [markdown]
# # Step 1: Data acquisition (Lahore PM2.5)
# Downloads measured PM2.5 (OpenAQ), weather (Open-Meteo ERA5) and the CAMS forecast
# (Open-Meteo), then merges them on one hourly grid in Pakistan time.
#
# Run from the repository root:  python notebooks/01_data_acquisition.py
# Or in Colab, one "# %%" block per cell, after cloning the repo and `%cd pm25-lahore`.
#
# The OpenAQ key is read by src/config.py from Colab Secrets or the .env file.
# It is never printed or saved by this script.

# %%
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path.cwd()))          # repo root on the path (works in Colab too)
from src.config import (CAMS_START, DATA_MERGED, DATA_RAW, EVENT_THRESHOLD, LAT, LON,
                        SEARCH_RADIUS_M, TZ, get_secret)

DATA_RAW.mkdir(parents=True, exist_ok=True)
OPENAQ = "https://api.openaq.org/v3"
SESSION = requests.Session()
SESSION.headers["X-API-Key"] = get_secret("OPENAQ_API_KEY")


def openaq_get(path, params):
    """GET from OpenAQ with retry on rate limit. Never logs headers (they hold the key)."""
    for attempt in range(5):
        r = SESSION.get(f"{OPENAQ}{path}", params=params, timeout=120)
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code == 401:
            raise RuntimeError("OpenAQ rejected the key (401). Check OPENAQ_API_KEY.")
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"OpenAQ kept rate-limiting {path}")


# %% [markdown]
# ## 1. Which PM2.5 stations exist near Lahore, and for how long?

# %%
def find_pm25_sensors(coord):
    data = openaq_get("/locations", {"coordinates": coord, "radius": SEARCH_RADIUS_M, "limit": 100})
    rows = []
    for loc in data.get("results", []):
        for s in loc.get("sensors", []):
            if s["parameter"]["name"] == "pm25":
                rows.append({
                    "location": loc.get("name"),
                    "location_id": loc["id"],
                    "sensor_id": s["id"],
                    "provider": (loc.get("provider") or {}).get("name"),
                    "first": (loc.get("datetimeFirst") or {}).get("utc"),
                    "last": (loc.get("datetimeLast") or {}).get("utc"),
                })
    return rows

rows = find_pm25_sensors(f"{LAT},{LON}") or find_pm25_sensors(f"{LON},{LAT}")
if not rows:
    raise SystemExit("No PM2.5 sensors found near Lahore. Send this message to the team.")

sensors = pd.DataFrame(rows)
sensors["first"] = pd.to_datetime(sensors["first"], utc=True)
sensors["last"] = pd.to_datetime(sensors["last"], utc=True)
sensors["years"] = ((sensors["last"] - sensors["first"]).dt.days / 365).round(1)
sensors = sensors.sort_values("years", ascending=False).reset_index(drop=True)
sensors.to_csv(DATA_RAW / "openaq_lahore_sensors.csv", index=False)
print(sensors.to_string())

# %% [markdown]
# ## 2. Download hourly PM2.5 for one station
# Default: the longest record. To pick another, change ROW to its row number above.

# %%
ROW = 0
SENSOR_ID = int(sensors.loc[ROW, "sensor_id"])
start = max(sensors.loc[ROW, "first"], pd.Timestamp(CAMS_START, tz="UTC") - pd.Timedelta(days=400))
end = sensors.loc[ROW, "last"]
print(f"Sensor {SENSOR_ID} ({sensors.loc[ROW, 'location']}): {start:%Y-%m-%d} to {end:%Y-%m-%d}")

records, page = [], 1
while True:
    data = openaq_get(f"/sensors/{SENSOR_ID}/hours", {
        "datetime_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datetime_to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 1000, "page": page,
    })
    res = data.get("results", [])
    records += [{"time_utc": m["period"]["datetimeFrom"]["utc"], "pm25": m["value"]} for m in res]
    print(f"page {page}: {len(res)} rows (total {len(records)})")
    if len(res) < 1000:
        break
    page += 1
    time.sleep(1)

pm = pd.DataFrame(records)
pm["time"] = (pd.to_datetime(pm["time_utc"], utc=True).dt.tz_convert(TZ)
              .dt.tz_localize(None).dt.floor("h"))
pm = pm.groupby("time", as_index=False)["pm25"].mean()
pm.to_csv(DATA_RAW / "openaq_pm25.csv", index=False)
print(pm["pm25"].describe())

# %% [markdown]
# ## 3. Weather (ERA5) and CAMS from Open-Meteo for the same range (no key needed)

# %%
d0 = max(pm["time"].min(), pd.Timestamp(CAMS_START)).strftime("%Y-%m-%d")
d1 = min(pm["time"].max(), pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
print("Open-Meteo range:", d0, "to", d1)

w = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": LAT, "longitude": LON, "start_date": d0, "end_date": d1, "timezone": TZ,
    "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,"
              "surface_pressure,wind_speed_10m,wind_direction_10m,boundary_layer_height",
}, timeout=300).json()
if "hourly" not in w:
    raise SystemExit(f"ERA5 request failed: {w}")
era5 = pd.DataFrame(w["hourly"])
era5["time"] = pd.to_datetime(era5["time"])
era5.to_csv(DATA_RAW / "era5_weather.csv", index=False)

c = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
    "latitude": LAT, "longitude": LON, "start_date": d0, "end_date": d1, "timezone": TZ,
    "hourly": "pm2_5,pm10", "domains": "cams_global",
}, timeout=300).json()
if "hourly" not in c:
    raise SystemExit(f"CAMS request failed: {c}")
cams = pd.DataFrame(c["hourly"]).rename(columns={"pm2_5": "cams_pm25", "pm10": "cams_pm10"})
cams["time"] = pd.to_datetime(cams["time"])
cams.to_csv(DATA_RAW / "cams_pm25.csv", index=False)
print(f"ERA5 rows: {len(era5)} | CAMS rows: {len(cams)} | "
      f"CAMS non-null: {cams['cams_pm25'].notna().sum()}")

# %% [markdown]
# ## 4. Merge on one hourly grid and report what we have

# %%
grid = pd.DataFrame({"time": pd.date_range(d0, f"{d1} 23:00", freq="h")})
df = (grid.merge(pm, on="time", how="left")
          .merge(era5, on="time", how="left")
          .merge(cams[["time", "cams_pm25"]], on="time", how="left"))
df.to_csv(DATA_MERGED, index=False)

roll24 = df["pm25"].rolling(24, min_periods=18).mean()
print(f"Rows: {len(df)} | {df.time.min()} to {df.time.max()}")
print(f"Missing PM2.5 hours: {100 * df['pm25'].isna().mean():.1f}%")
print(f"Missing CAMS hours:  {100 * df['cams_pm25'].isna().mean():.1f}%")
print(f"Hours with 24-h mean > {EVENT_THRESHOLD}: {100 * (roll24 > EVENT_THRESHOLD).mean():.1f}%")
print(df.dropna().head(10).to_string(index=False))
