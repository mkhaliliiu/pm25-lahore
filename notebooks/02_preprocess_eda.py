# %% [markdown]
# # Step 2: Diagnose, clean, build features, and explore (EDA)
# Run after 01_data_acquisition.py:   %run notebooks/02_preprocess_eda.py
# Outputs: data/features.csv, results/data_summary.json, results/figures/*.png

# %%
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from src.config import CAMS_START, EVENT_THRESHOLD, RESULTS, ROOT
from src.features import (add_features, add_targets, clean_pm25, load_openaq_chunks,
                          load_weather_and_cams, to_hourly_grid)

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
summary = {}

# %% [markdown]
# ## 1. Diagnose the raw OpenAQ download (duplicates, timestamps)

# %%
pm_raw, rep = load_openaq_chunks(sensor_id=25135)
summary["download"] = rep
print(json.dumps(rep, indent=2))

# %% [markdown]
# ## 2. Clean and put everything on one hourly grid

# %%
pm_clean, rep = clean_pm25(pm_raw)
summary["cleaning"] = rep
era5, cams = load_weather_and_cams()
df = to_hourly_grid(pm_clean, era5, cams)

summary["grid"] = {
    "start": str(df.index.min()), "end": str(df.index.max()), "hours": len(df),
    "observed_pm25_hours": int(df["pm25_obs"].notna().sum()),
    "missing_pm25_pct": round(100 * df["pm25_obs"].isna().mean(), 1),
    "hours_filled_by_interpolation": int(df["pm25"].notna().sum() - df["pm25_obs"].notna().sum()),
}
print(json.dumps(summary["cleaning"], indent=2))
print(json.dumps(summary["grid"], indent=2))
print("\nObserved PM2.5 hours per year:")
print(df["pm25_obs"].notna().groupby(df.index.year).sum().to_string())

# %% [markdown]
# ## 3. Features and targets

# %%
feat = add_targets(add_features(df))
feat.to_csv(ROOT / "data" / "features.csv", index_label="time")

valid = feat["mean24_obs"].notna()
event = feat.loc[valid, "mean24_obs"] > EVENT_THRESHOLD
season = np.where(feat.index[valid].month.isin([10, 11, 12, 1, 2]), "smog season (Oct-Feb)", "rest of year")
summary["events"] = {
    "event_rate_pct": round(100 * event.mean(), 1),
    "imbalance_ratio": f"{(1 - event.mean()) / event.mean():.1f}:1",
    "event_rate_by_season_pct": (100 * event.groupby(season).mean()).round(1).to_dict(),
}
print(json.dumps(summary["events"], indent=2))

# %% [markdown]
# ## 4. EDA figures (saved to results/figures for the paper)

# %%
obs = feat["pm25_obs"]

# 4a. Monthly mean over time
fig, ax = plt.subplots(figsize=(10, 3.5))
obs.resample("MS").mean().plot(ax=ax, color="#1f4e79")
ax.axhline(EVENT_THRESHOLD, ls="--", color="#c0392b", lw=1, label="Very Unhealthy (125.5)")
ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_xlabel(""); ax.set_title("Monthly mean PM2.5, US Diplomatic Post Lahore")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "monthly_mean.png", dpi=200); plt.close(fig)

# 4b. Seasonality: distribution by month
fig, ax = plt.subplots(figsize=(10, 3.5))
data = [obs[obs.index.month == m].dropna() for m in range(1, 13)]
ax.boxplot(data, showfliers=False)
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.axhline(EVENT_THRESHOLD, ls="--", color="#c0392b", lw=1)
ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_title("PM2.5 by month (outliers hidden)")
fig.tight_layout(); fig.savefig(FIG / "monthly_boxplot.png", dpi=200); plt.close(fig)

# 4c. Diurnal cycle, smog season vs rest
fig, ax = plt.subplots(figsize=(6, 3.5))
smog = obs.index.month.isin([10, 11, 12, 1, 2])
obs[smog].groupby(obs[smog].index.hour).mean().plot(ax=ax, label="Oct-Feb", color="#c0392b")
obs[~smog].groupby(obs[~smog].index.hour).mean().plot(ax=ax, label="Mar-Sep", color="#1f4e79")
ax.set_xlabel("Hour of day (PKT)"); ax.set_ylabel("Mean PM2.5 (µg/m³)"); ax.set_title("Diurnal cycle")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "diurnal_cycle.png", dpi=200); plt.close(fig)

# 4d. Weather relationships
cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "boundary_layer_height",
        "log_ventilation", "precipitation", "surface_pressure"]
corr = feat[["pm25_obs"] + cols].corr(method="spearman")["pm25_obs"].drop("pm25_obs").sort_values()
summary["spearman_with_pm25"] = corr.round(2).to_dict()
fig, ax = plt.subplots(figsize=(6, 3.5))
corr.plot.barh(ax=ax, color=["#c0392b" if v < 0 else "#1f4e79" for v in corr])
ax.set_xlabel("Spearman correlation with PM2.5"); ax.set_title("Weather drivers")
fig.tight_layout(); fig.savefig(FIG / "weather_correlation.png", dpi=200); plt.close(fig)

# 4e. CAMS vs observed (short window)
short = feat.loc[feat.index >= CAMS_START, ["pm25_obs", "cams_pm25"]].dropna()
bias = (short["cams_pm25"] - short["pm25_obs"])
summary["cams_vs_observed"] = {
    "hours": len(short),
    "pearson_r": round(short.corr().iloc[0, 1], 2),
    "mean_bias": round(bias.mean(), 1),
    "mae": round(bias.abs().mean(), 1),
    "mean_bias_smog_season": round(bias[short.index.month.isin([10, 11, 12, 1, 2])].mean(), 1),
}
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(short["pm25_obs"], short["cams_pm25"], s=2, alpha=0.2, color="#1f4e79")
lim = [0, np.nanpercentile(short.values, 99.5)]
ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed PM2.5"); ax.set_ylabel("CAMS PM2.5"); ax.set_title("CAMS vs observed (hourly)")
fig.tight_layout(); fig.savefig(FIG / "cams_vs_observed.png", dpi=200); plt.close(fig)
print(json.dumps(summary["cams_vs_observed"], indent=2))

# %%
with open(RESULTS / "data_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("Saved:", sorted(p.name for p in FIG.glob("*.png")))
print(json.dumps(summary, indent=2, default=str))
