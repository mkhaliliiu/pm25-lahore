# %% [markdown]
# # Step 5d: Remaining proposal items
# A. RQ3 as worded in the proposal: recall at a fixed precision (plain vs class weights vs SMOTE).
# B. Section 8.5: classification metrics as mean ± standard deviation over folds.
# C. Section 8.4: false alarms per month counted as days and as episodes (not only hours).
# D. Section 7 / objective 1: seaborn month x hour heatmap and statsmodels autocorrelation
#    (seasonal and diurnal patterns, and the choice of lags 1, 24 and 48 h).
# E. Section 5: "multiple stations combined where available to decrease gaps". Other OpenAQ
#    sensors that ran alongside the US Embassy monitor are calibrated to it (on data before the
#    test period only) and used to fill gaps in the PM2.5 *inputs*. The backtest is repeated with
#    the filled inputs; targets stay the Embassy measurements.
#
# Run after 03 and 04:   %run notebooks/09_proposal_checks.py   (A-D: 1 min; E: 5-15 min, needs the
# OpenAQ key, resumable). Outputs: results/rq3_recall_at_precision.csv, classification_fold_stats.csv,
# false_alarms_per_month.csv, multistation_*.csv/json, figures/heatmap_month_hour.png, figures/acf_pm25.png

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
import requests
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from statsmodels.graphics.tsaplots import plot_acf

sys.path.insert(0, str(Path.cwd()))
from src.config import CAMS_START, HORIZONS, LAT, LON, RESULTS, ROOT, SEARCH_RADIUS_M, get_secret
from src.evaluation import classification_metrics, diebold_mariano, make_folds, regression_metrics, split_masks
from src.features import add_features, add_targets, feature_list
from src.live import openaq_hours
from src.models import lgbm_regressor, ridge_regressor

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FIRST_TEST = pd.Timestamp("2023-08-01")
COST = 10
VARIANTS = ["LightGBM plain", "LightGBM class weights", "LightGBM SMOTE"]
pd.set_option("display.width", 220)

cls = pd.read_csv(ROOT / "data" / "predictions_classification.csv", parse_dates=["time"], index_col="time")
feat = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"], index_col="time")

# %% [markdown]
# ## A. Recall at a fixed precision (RQ3: "better for recall ... for a given precision")

# %%
rows = []
for (ev, h), g in cls.groupby(["event", "h"]):
    for m in ["Persistence", "Raw CAMS"] + VARIANTS:
        p, r, _ = precision_recall_curve(g["y"], g[f"score|{m}"])
        for target in (0.80, 0.90, 0.95):
            ok = p >= target
            rows.append({"event": ev, "h": h, "model": m, "precision_at_least": target,
                         "recall": float(r[ok].max()) if ok.any() else 0.0})
rap = pd.DataFrame(rows)
rap.round(3).to_csv(RESULTS / "rq3_recall_at_precision.csv", index=False)
print("Recall at a given precision (pooled out-of-sample scores):")
print(rap.pivot_table(index=["event", "h", "model"], columns="precision_at_least", values="recall").round(3).to_string())

# %% [markdown]
# ## B. Classification metrics per fold: mean ± standard deviation (folds with at least one event)

# %%
rows = []
for (ev, h, fold), g in cls.groupby(["event", "h", "fold"]):
    for m in ["Persistence", "Raw CAMS"] + VARIANTS:
        met = classification_metrics(g["y"], g[f"score|{m}"], fn_cost=COST, pred=g[f"pred{COST}|{m}"])
        rows.append({"event": ev, "h": h, "fold": fold, "model": m, **met})
pf = pd.DataFrame(rows)
with_events = pf[pf.positives > 0]
stats = with_events.groupby(["event", "h", "model"])[["recall", "precision", "F1", "PR_AUC", "false_alarm_hours_per_30d"]] \
    .agg(["mean", "std"]).round(3)
stats.columns = [f"{a}_{b}" for a, b in stats.columns]
stats["folds_with_events"] = with_events.groupby(["event", "h", "model"]).size()
stats.to_csv(RESULTS / "classification_fold_stats.csv")
print("\nPer-fold mean ± std (cost 10:1):")
print(stats[[c for c in stats.columns if c.startswith(("recall", "PR_AUC", "folds"))]].to_string())

# %% [markdown]
# ## C. False alarms per month, counted as days and as episodes
# False-alarm day: a target day with at least one warning but no event hour.
# False-alarm episode: a run of consecutive false-alarm hours.

# %%
rows = []
for (ev, h), g in cls.groupby(["event", "h"]):
    g = g.sort_index()
    tt = g.index + pd.Timedelta(hours=int(h))
    months = len(pd.Index(tt.to_period("M")).unique())
    for m in ["Persistence", "Raw CAMS"] + VARIANTS:
        warn, y = g[f"pred{COST}|{m}"].to_numpy(), g["y"].to_numpy()
        daily = pd.DataFrame({"warn": warn, "event": y}, index=tt).groupby(tt.normalize()).max()
        fa_days = int(((daily.warn == 1) & (daily.event == 0)).sum())
        fa = pd.Series((warn == 1) & (y == 0), index=tt)
        gap = fa.index.to_series().diff() != pd.Timedelta(hours=1)
        episodes = int((fa & (~fa.shift(1, fill_value=False) | gap)).sum())
        rows.append({"event": ev, "h": h, "model": m, "test_months": months,
                     "false_alarm_days_per_month": fa_days / months,
                     "false_alarm_episodes_per_month": episodes / months,
                     "false_alarm_hours_per_month": int(fa.sum()) / months})
fam = pd.DataFrame(rows)
fam.round(2).to_csv(RESULTS / "false_alarms_per_month.csv", index=False)
print("\nFalse alarms per month (test months with data):")
print(fam[fam.event.str.startswith("Very")].round(2).to_string(index=False))

# %% [markdown]
# ## D. Seasonal-diurnal heatmap (seaborn) and autocorrelation (statsmodels)

# %%
obs = feat["pm25_obs"]
heat = obs.groupby([obs.index.month, obs.index.hour]).mean().unstack()
heat.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(heat)]
fig, ax = plt.subplots(figsize=(10, 4.2))
sns.heatmap(heat, cmap="rocket_r", ax=ax, cbar_kws={"label": "Mean PM2.5 (µg/m³)"})
ax.set_xlabel("Hour of day (PKT)"); ax.set_ylabel(""); ax.set_title("Mean PM2.5 by month and hour, US Embassy Lahore 2019-2025")
fig.tight_layout(); fig.savefig(FIG / "heatmap_month_hour.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.4))
plot_acf(feat["pm25"], lags=168, ax=ax, missing="conservative",
         markersize=2, title="Autocorrelation of hourly PM2.5 (lags up to 1 week)")
for lag in (24, 48, 168):
    ax.axvline(lag, color="#8a8985", lw=0.8, ls="--")
ax.set_xlabel("Lag (hours)")
fig.tight_layout(); fig.savefig(FIG / "acf_pm25.png", dpi=200); plt.close(fig)
print("\nSaved heatmap_month_hour.png and acf_pm25.png")

# %% [markdown]
# ## E. Other stations that ran alongside the Embassy monitor: calibrate and fill input gaps

# %%
KEY = get_secret("OPENAQ_API_KEY")
S = requests.Session()
S.headers["X-API-Key"] = KEY
EMB0, EMB1 = feat.index.min(), feat.index.max()


def locations(coord):
    r = S.get("https://api.openaq.org/v3/locations", timeout=60,
              params={"coordinates": coord, "radius": SEARCH_RADIUS_M, "limit": 1000})
    r.raise_for_status()
    return r.json().get("results", [])


rows = []
for loc in locations(f"{LAT},{LON}") or locations(f"{LON},{LAT}"):
    first = pd.to_datetime((loc.get("datetimeFirst") or {}).get("utc"), utc=True)
    last = pd.to_datetime((loc.get("datetimeLast") or {}).get("utc"), utc=True)
    for s in loc.get("sensors", []):
        if s["parameter"]["name"] == "pm25" and s["id"] != 25135:
            rows.append({"location": loc["name"], "sensor_id": s["id"], "provider": (loc.get("provider") or {}).get("name"),
                         "is_monitor": loc.get("isMonitor"), "first": first, "last": last})
hist = pd.DataFrame(rows)
for c in ("first", "last"):
    hist[c] = hist[c].dt.tz_convert("Asia/Karachi").dt.tz_localize(None)
hist["overlap_days"] = ((hist["last"].clip(upper=EMB1) - hist["first"].clip(lower=EMB0)).dt.days).clip(lower=0)
hist = hist.sort_values("overlap_days", ascending=False)
hist.to_csv(RESULTS / "multistation_candidates.csv", index=False)
print(f"{len(hist)} other PM2.5 sensors within 25 km; {int((hist.overlap_days >= 60).sum())} overlap the "
      f"Embassy record by 60+ days")
print(hist.head(25).to_string(index=False))

cand = hist[(hist.overlap_days >= 60) & (hist["first"] < FIRST_TEST)]
summary = {"candidates_overlapping_60_days": int((hist.overlap_days >= 60).sum()),
           "candidates_starting_before_test_period": len(cand)}

# %%
if len(cand):
    CH = ROOT / "data" / "raw" / "history_chunks"
    CH.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(cand.itertuples()):
        for m0 in pd.date_range(max(r.first, EMB0).replace(day=1).normalize(), min(r.last, EMB1), freq="MS"):
            out = CH / f"{r.sensor_id}_{m0:%Y_%m}.csv"
            if out.exists():
                continue
            x = openaq_hours(KEY, r.sensor_id, m0 - pd.Timedelta(hours=5),
                             m0 + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=5))
            if x is None:
                continue
            x.rename("value").to_csv(out, index_label="time")
            time.sleep(1.1)
        print(f"{i + 1}/{len(cand)} sensors downloaded")

    # per-sensor linear calibration to the Embassy monitor, fitted only before the test period
    filled_parts, cal_rows = [], []
    for r in cand.itertuples():
        parts = [pd.read_csv(f, parse_dates=["time"], index_col="time")["value"] for f in sorted(CH.glob(f"{r.sensor_id}_*.csv"))]
        parts = [p for p in parts if len(p)]
        if not parts:
            continue
        x = pd.concat(parts).groupby(level=0).median()
        x = x[(x > 0) & (x < 1500)]
        both = pd.DataFrame({"x": x, "emb": feat["pm25_obs"]}).dropna()
        fit = both[both.index < FIRST_TEST - pd.Timedelta(hours=72)]
        if len(fit) < 200:
            continue
        rcorr = fit.corr().iloc[0, 1]
        slope, intercept = np.polyfit(fit.x, fit.emb, 1)
        cal_rows.append({"sensor_id": r.sensor_id, "location": r.location, "provider": r.provider,
                         "fit_hours": len(fit), "r": round(rcorr, 3), "slope": round(slope, 3), "intercept": round(intercept, 1),
                         "used": bool(rcorr >= 0.7)})
        if rcorr >= 0.7:
            filled_parts.append((intercept + slope * x).clip(lower=1).rename(r.sensor_id))
    cal_table = pd.DataFrame(cal_rows)
    cal_table.to_csv(RESULTS / "multistation_calibration.csv", index=False)
    print("\nPer-sensor calibration (fit before Aug 2023; used if r >= 0.7):")
    print(cal_table.to_string(index=False) if len(cal_table) else "  none had 200+ common hours")
    summary["sensors_used_for_filling"] = len(filled_parts)
else:
    filled_parts = []
    print("\nNo other sensor overlaps the Embassy record before the test period; nothing to combine.")
    summary["sensors_used_for_filling"] = 0

# %% [markdown]
# ### E2. Backtest with gap-filled inputs (Ridge + CAMS and LightGBM + CAMS, same folds and test hours)

# %%
if filled_parts:
    fill = pd.concat(filled_parts, axis=1).median(axis=1).reindex(feat.index)
    filled = feat.copy()
    missing_before = int(filled["pm25"].isna().sum())
    obs_filled = filled["pm25_obs"].fillna(fill)
    filled["pm25"] = obs_filled.interpolate(limit=3, limit_area="inside")
    derived = [c for c in filled.columns if c.startswith("pm25_") and c not in ("pm25_obs",)]
    filled = add_targets(add_features(filled.drop(columns=derived)))   # targets still from pm25_obs (Embassy)
    summary["input_hours_missing_before"] = missing_before
    summary["input_hours_missing_after"] = int(filled["pm25"].isna().sum())

    base = add_targets(feat)
    reg = pd.read_csv(ROOT / "data" / "predictions_regression.csv", parse_dates=["time"], index_col="time")
    folds = make_folds(feat.index[feat.index >= CAMS_START], FIRST_TEST)
    out_rows, dm_rows = [], []
    for h in HORIZONS:
        y, cols = base[f"y_pm25_h{h}"], feature_list(h, True)
        preds = []
        for k, (a, b) in enumerate(folds):
            tr, _ = split_masks(feat.index, a, b, h, train_start=CAMS_START)
            tr &= y.notna()
            te = reg[(reg.h == h) & (reg.fold == k)].index
            if len(te) == 0:
                continue
            p = pd.DataFrame(index=te)
            p["y"] = y[te]
            for name, make in [("Ridge + CAMS", ridge_regressor), ("LightGBM + CAMS", lgbm_regressor)]:
                p[name] = np.maximum(make().fit(base.loc[tr, cols], y[tr]).predict(base.loc[te, cols]), 0)
                p[name + " (gap-filled inputs)"] = np.maximum(
                    make().fit(filled.loc[tr, cols], y[tr]).predict(filled.loc[te, cols]), 0)
            preds.append(p)
        P = pd.concat(preds).sort_index()
        for name in ["Ridge + CAMS", "LightGBM + CAMS"]:
            for variant in [name, name + " (gap-filled inputs)"]:
                out_rows.append({"h": h, "model": variant, **regression_metrics(P["y"], P[variant])})
            stat, pv = diebold_mariano(P[name + " (gap-filled inputs)"] - P["y"], P[name] - P["y"], h)
            dm_rows.append({"h": h, "model": name, "DM_filled_vs_original": stat, "p_value": pv})
    ms = pd.DataFrame(out_rows)
    ms.round(3).to_csv(RESULTS / "multistation_backtest.csv", index=False)
    pd.DataFrame(dm_rows).round(4).to_csv(RESULTS / "multistation_dm.csv", index=False)
    print("\nBacktest with original vs gap-filled PM2.5 inputs:")
    print(ms.round(2).to_string(index=False))
    print(pd.DataFrame(dm_rows).round(3).to_string(index=False))

(RESULTS / "multistation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("\n", json.dumps(summary, indent=2, default=str))
