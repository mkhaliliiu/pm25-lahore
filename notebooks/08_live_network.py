# %% [markdown]
# # Step 5c: The live sensor network (OpenAQ), calibration to the reference monitor, hourly models
# Lahore has ~50 live low-cost PM2.5 sensors on OpenAQ (AirGradient, Hawanama) but no live reference
# monitor. This notebook:
#   1. downloads the sensors' history for Nov 2024 - Mar 2025, when the US Embassy monitor still ran;
#   2. compares the network's hourly median with the Embassy monitor and fits a linear calibration;
#   3. tests the deployed pipeline honestly: the full model fed with network inputs instead of
#      Embassy inputs, scored against the Embassy monitor;
#   4. chooses the sensors the live app will read, and trains Ridge models for every horizon 1-72 h.
#
# Needs data/raw/openaq_lahore_active_sensors.csv (from the sensor check cell) and the OpenAQ key.
# Run:   %run notebooks/08_live_network.py   (about 10 min: ~270 small OpenAQ requests, resumable)
# Outputs: models/hourly_full.json, models/hourly_live.json, models/meta_hourly.json,
#   app/data/live_sensors.csv, results/network_calibration.json, results/network_pipeline_test.csv,
#   results/figures/network_vs_embassy.png

# %%
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from src.config import CAMS_START, RESULTS, ROOT, get_secret
from src.deploy import MODEL_DIR, ridge_to_dict
from src.evaluation import regression_metrics
from src.features import (FEATURES_BASE, LIVE_BASE, add_cams_horizons, add_features, add_live_features,
                          add_targets)
from src.live import HOURLY, network_median, openaq_hours
from src.models import ridge_regressor

KEY = get_secret("OPENAQ_API_KEY")
OV0, OV1 = pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-05")    # overlap with the Embassy monitor
MIN_SENSORS = 3
N_LIVE = 15
APP_DATA, CHUNKS = ROOT / "app" / "data", ROOT / "data" / "raw" / "network_chunks"
CHUNKS.mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

active = pd.read_csv(ROOT / "data" / "raw" / "openaq_lahore_active_sensors.csv")
print(f"{len(active)} active sensors: {active.provider.value_counts().to_dict()}")

# %% [markdown]
# ## 1. Download the overlap period, one sensor-month per file (re-running skips finished files)

# %%
months = pd.date_range(OV0, OV1, freq="MS")
for i, sid in enumerate(active.sensor_id):
    for m0 in months:
        out = CHUNKS / f"{sid}_{m0:%Y_%m}.csv"
        if out.exists():
            continue
        m1 = min(m0 + pd.offsets.MonthBegin(1), OV1)
        x = openaq_hours(KEY, sid, m0 - pd.Timedelta(hours=5), m1 - pd.Timedelta(hours=5))
        if x is None:
            print(f"  sensor {sid} {m0:%Y-%m}: failed, will retry on the next run")
            continue
        x.rename("value").to_csv(out, index_label="time")
        time.sleep(1.1)                                  # stay under OpenAQ's 60 requests/minute
    print(f"{i + 1}/{len(active)} sensors done")

series = {}
for sid in active.sensor_id:
    parts = [pd.read_csv(f, parse_dates=["time"], index_col="time")["value"]
             for f in sorted(CHUNKS.glob(f"{sid}_*.csv"))]
    parts = [p for p in parts if len(p)]
    if parts:
        series[sid] = pd.concat(parts).groupby(level=0).median()
wide = pd.DataFrame(series)
active["overlap_hours"] = active.sensor_id.map(wide.notna().sum()).fillna(0).astype(int)
print(active[["location", "provider", "hours_in_72h", "overlap_hours"]].sort_values(
    "overlap_hours", ascending=False).to_string(index=False))

# %% [markdown]
# ## 2. Choose the live sensors and calibrate their median to the Embassy monitor

# %%
cand = active[active.hours_in_72h >= 60].sort_values(["overlap_hours", "hours_in_72h"], ascending=False)
live = cand.head(N_LIVE)
live[["sensor_id", "location", "provider"]].to_csv(APP_DATA / "live_sensors.csv", index=False)
print(f"\nLive sensors ({len(live)}):", ", ".join(live.location))

df = add_live_features(add_targets(pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"],
                                               index_col="time")))
cols = [c for c in live.sensor_id if c in wide.columns]
net, n = network_median(wide[cols], MIN_SENSORS) if cols else (pd.Series(dtype=float), None)
both = pd.DataFrame({"network": net, "embassy": df["pm25_obs"]}).dropna()
both = both[(both.index >= OV0) & (both.index < OV1)]

cal = {"slope": 1.0, "intercept": 0.0, "hours": len(both), "note": "identity (too little overlap)"}
if len(both) >= 300:
    slope, intercept = np.polyfit(both["network"], both["embassy"], 1)
    fitted = intercept + slope * both["network"]
    cal = {"slope": round(float(slope), 4), "intercept": round(float(intercept), 2), "hours": len(both),
           "pearson_r": round(float(both.corr().iloc[0, 1]), 3),
           "mean_network": round(float(both.network.mean()), 1), "mean_embassy": round(float(both.embassy.mean()), 1),
           "mae_raw": round(float((both.network - both.embassy).abs().mean()), 1),
           "mae_calibrated": round(float((fitted - both.embassy).abs().mean()), 1),
           "note": "embassy = intercept + slope x network median (Nov 2024 - Mar 2025)"}
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.scatter(both.network, both.embassy, s=3, alpha=0.25, color="#2a78d6")
    lim = [0, np.nanpercentile(both.values, 99.5)]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.plot(lim, [intercept + slope * v for v in lim], color="#eb6834", lw=2, label="calibration")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.legend()
    ax.set_xlabel("Low-cost network median (µg/m³)"); ax.set_ylabel("US Embassy monitor (µg/m³)")
    ax.set_title(f"Hourly, Nov 2024 - Mar 2025 (r = {cal['pearson_r']})")
    fig.tight_layout(); fig.savefig(RESULTS / "figures" / "network_vs_embassy.png", dpi=200); plt.close(fig)
(RESULTS / "network_calibration.json").write_text(json.dumps(cal, indent=2))
print("\nCalibration:", json.dumps(cal, indent=2))

# %% [markdown]
# ## 3. Deployment-realistic test (only if the overlap is long enough)
# Train on data before Nov 2024 with Embassy inputs; predict Nov 2024 - Mar 2025 from (a) Embassy
# inputs, (b) calibrated network inputs, (c) no measurements (live model), and compare with raw CAMS
# and persistence. The target is always the Embassy monitor.

# %%
if len(both) >= 300:
    netdf = df.copy()
    netdf["pm25_obs"] = (cal["intercept"] + cal["slope"] * net).reindex(df.index).clip(lower=1)
    netdf.loc[(netdf.index < OV0) | (netdf.index >= OV1), "pm25_obs"] = np.nan
    netdf["pm25"] = netdf["pm25_obs"].interpolate(limit=3, limit_area="inside")
    netdf = add_live_features(add_features(netdf.drop(columns=[c for c in netdf.columns if c.startswith("pm25_")
                                                               and c != "pm25_obs"])))
    rows = []
    for h in (24, 48, 72):
        y = df[f"y_pm25_h{h}"]
        full_cols = FEATURES_BASE + [f"cams_h{h}", f"cams_mean24_h{h}"]
        live_cols = LIVE_BASE + [f"cams_h{h}", f"cams_mean24_h{h}"]
        tr = (df.index >= CAMS_START) & (df.index < OV0 - pd.Timedelta(hours=h)) & y.notna()
        te = (df.index >= OV0) & (df.index < OV1) & y.notna() & df["pm25"].notna() & netdf["pm25"].notna()
        full = ridge_regressor().fit(df.loc[tr, full_cols], y[tr])
        livem = ridge_regressor().fit(df.loc[tr, live_cols], y[tr])
        preds = {"Full model, Embassy inputs": full.predict(df.loc[te, full_cols]),
                 "Full model, network inputs (deployed)": full.predict(netdf.loc[te, full_cols]),
                 "Live model, no measurements": livem.predict(df.loc[te, live_cols]),
                 "Persistence, network": netdf.loc[te, "pm25"], "Raw CAMS": df.loc[te, f"cams_h{h}"]}
        for name, p in preds.items():
            rows.append({"h": h, "model": name, **regression_metrics(y[te], np.maximum(p, 0))})
    test = pd.DataFrame(rows)
    test.round(3).to_csv(RESULTS / "network_pipeline_test.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nDeployment-realistic test, Nov 2024 - Mar 2025 (target: Embassy monitor)")
    print(test.round(2).to_string(index=False))
else:
    print("\nNot enough overlap with the Embassy monitor for the pipeline test; calibration = identity.")

# %% [markdown]
# ## 4. Ridge models for every horizon 1-72 h (full and live), for the hour-by-hour live forecast

# %%
short = add_cams_horizons(df[df.index >= CAMS_START], HOURLY)
models = {"full": {}, "live": {}}
for h in HOURLY:
    y = short["pm25_obs"].shift(-h)
    for kind, base in [("full", FEATURES_BASE), ("live", LIVE_BASE)]:
        cols_h = base + [f"cams_h{h}", f"cams_mean24_h{h}"]
        ok = y.notna()
        models[kind][h] = ridge_to_dict(ridge_regressor().fit(short.loc[ok, cols_h], y[ok]), cols_h)
for kind in models:
    (MODEL_DIR / f"hourly_{kind}.json").write_text(json.dumps(models[kind]))
meta = {"calibration": cal, "min_sensors": 4, "horizons": [1, 72],
        "sensors_file": "app/data/live_sensors.csv",
        "note": "hourly Ridge models; 24/48/72-h warnings come from meta.json / meta_live.json"}
(MODEL_DIR / "meta_hourly.json").write_text(json.dumps(meta, indent=2))
for f in ["hourly_full.json", "hourly_live.json", "meta_hourly.json"]:
    print(f"models/{f}: {(MODEL_DIR / f).stat().st_size / 1024:.0f} KB")

# %% [markdown]
# ## 5. Try the live pipeline now

# %%
from src.deploy import HourlyRidge, us_aqi
from src.live import build_live
res = build_live(KEY)
row = res["row"]
mode = "full model (anchored on live sensors)" if res["anchored"] else "live model (weather + CAMS only)"
print(f"\nIssued {res['issue']:%Y-%m-%d %H:%M} PKT | {mode} | sensors responding "
      f"{res['sensors_responding']}/{res['sensors_total']}")
if res["anchored"]:
    pm_now = row["pm25_obs"].iloc[0]
    print(f"Measured now (calibrated network median): {pm_now:.0f} µg/m³ = US AQI {us_aqi(pm_now)} "
          f"(raw median {row['pm25_raw_network'].iloc[0]:.0f}, {int(row['n_sensors'].iloc[0])} sensors)")
hr = HourlyRidge("full" if res["anchored"] else "live").predict(row).iloc[0]
for h in (6, 12, 24, 48, 72):
    print(f"  +{h:2d} h: {hr[h]:6.1f} µg/m³ (US AQI {us_aqi(hr[h])}) | raw CAMS {row[f'cams_h{h}'].iloc[0]:6.1f}")
