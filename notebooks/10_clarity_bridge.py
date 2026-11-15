# %% [markdown]
# # Step 5e: Calibrating the live network through a bridge sensor, and combining stations
# The live low-cost sensors started after the US Embassy monitor stopped, so they cannot be compared
# with it directly. One Clarity sensor in Lahore (OpenAQ sensor 7466365) overlapped both:
#   leg 1: Clarity vs Embassy        (Nov 2023 - Mar 2025)
#   leg 2: live network vs Clarity   (May 2025 - Sep 2026)
# Chaining the two linear fits converts the live network median to the Embassy (training) scale.
#
# The same sensor is used for proposal Section 5 ("multiple stations combined where available to
# decrease gaps"): in each backtest fold it is calibrated to the Embassy monitor on training data only
# and fills gaps in the PM2.5 inputs. Targets stay the Embassy measurements.
#
# Run after 08:   %run notebooks/10_clarity_bridge.py   (about 10 min, resumable; needs the OpenAQ key)
# Outputs: models/meta_hourly.json (calibration), results/bridge_calibration.json,
#   results/multistation_backtest.csv, results/multistation_dm.csv, results/figures/bridge_calibration.png

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
from src.config import CAMS_START, HORIZONS, RESULTS, ROOT, get_secret
from src.evaluation import diebold_mariano, make_folds, regression_metrics, split_masks
from src.features import add_features, add_targets, feature_list
from src.live import network_median, openaq_hours
from src.models import lgbm_regressor, ridge_regressor

KEY = get_secret("OPENAQ_API_KEY")
CLARITY = 7466365
FIRST_TEST = pd.Timestamp("2023-08-01")
MIN_PAIRS, MIN_R = 500, 0.7
CH = ROOT / "data" / "raw" / "bridge_chunks"
CH.mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)


def download(sid, start, end):
    """Monthly chunks of one sensor (Pakistan-time months), skipping files already on disk."""
    for m0 in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        out = CH / f"{sid}_{m0:%Y_%m}.csv"
        if out.exists():
            continue
        x = openaq_hours(KEY, sid, m0 - pd.Timedelta(hours=5), m0 + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=5))
        if x is None:
            print(f"  {sid} {m0:%Y-%m} failed; re-run to retry")
            continue
        x.rename("value").to_csv(out, index_label="time")
        time.sleep(1.1)
    parts = [pd.read_csv(f, parse_dates=["time"], index_col="time")["value"] for f in sorted(CH.glob(f"{sid}_*.csv"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).groupby(level=0).median()
    return s[(s > 0) & (s < 1500)]


def fit_line(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < MIN_PAIRS:
        return None
    slope, intercept = np.polyfit(d.x, d.y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "pairs": len(d),
            "r": float(d.corr().iloc[0, 1]), "mean_x": float(d.x.mean()), "mean_y": float(d.y.mean()),
            "start": str(d.index.min()), "end": str(d.index.max())}


feat = add_targets(pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"], index_col="time"))
live = pd.read_csv(ROOT / "app" / "data" / "live_sensors.csv")

# %% [markdown]
# ## 1. Download the bridge sensor and the live sensors' history

# %%
clarity = download(CLARITY, "2023-11-01", "2026-09-30")
print(f"Clarity sensor: {len(clarity)} hours, {clarity.index.min()} to {clarity.index.max()}")
net_parts = {}
for i, sid in enumerate(live.sensor_id):
    s = download(sid, "2025-05-01", "2026-09-30")
    if len(s):
        net_parts[sid] = s
    print(f"live sensor {i + 1}/{len(live)}: {len(s)} hours")
net = network_median(pd.DataFrame(net_parts), 4)[0] if net_parts else pd.Series(dtype=float)

# %% [markdown]
# ## 2. Two-leg calibration: network -> Clarity -> Embassy

# %%
leg1 = fit_line(clarity, feat["pm25_obs"])          # embassy = a1 + b1 * clarity
leg2 = fit_line(net, clarity)                        # clarity = a2 + b2 * network
print("Leg 1 (Embassy on Clarity):", json.dumps(leg1, indent=2) if leg1 else "too few common hours")
print("Leg 2 (Clarity on network):", json.dumps(leg2, indent=2) if leg2 else "too few common hours")

ok = bool(leg1 and leg2 and leg1["r"] >= MIN_R and leg2["r"] >= MIN_R)
if ok:
    slope = leg1["slope"] * leg2["slope"]
    intercept = leg1["intercept"] + leg1["slope"] * leg2["intercept"]
    cal = {"slope": round(slope, 4), "intercept": round(intercept, 2), "method": "bridge via Clarity sensor 7466365",
           "note": "embassy-scale = intercept + slope x live network median", "leg1": leg1, "leg2": leg2}
else:
    cal = {"slope": 1.0, "intercept": 0.0, "method": "none",
           "note": "bridge not reliable enough (needs r >= 0.7 and 500+ pairs on both legs)", "leg1": leg1, "leg2": leg2}
(RESULTS / "bridge_calibration.json").write_text(json.dumps(cal, indent=2))
meta_path = ROOT / "models" / "meta_hourly.json"
meta = json.loads(meta_path.read_text())
meta["calibration"] = {k: cal[k] for k in ("slope", "intercept", "method", "note")}
meta_path.write_text(json.dumps(meta, indent=2))
print(f"\nLive calibration written to models/meta_hourly.json: {meta['calibration']}")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
for ax, (x, y, leg, xl, yl) in zip(axes, [
        (clarity, feat["pm25_obs"], leg1, "Clarity sensor (µg/m³)", "US Embassy monitor (µg/m³)"),
        (net, clarity, leg2, "Live network median (µg/m³)", "Clarity sensor (µg/m³)")]):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d):
        ax.scatter(d.x, d.y, s=2, alpha=0.2, color="#2a78d6")
        lim = [0, np.nanpercentile(d.values, 99.5)]
        ax.plot(lim, lim, "k--", lw=1, label="1:1")
        if leg:
            ax.plot(lim, [leg["intercept"] + leg["slope"] * v for v in lim], color="#eb6834", lw=2,
                    label=f"fit (r = {leg['r']:.2f}, n = {leg['pairs']})")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.legend(fontsize=8)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
axes[0].set_title("Leg 1: Nov 2023 - Mar 2025"); axes[1].set_title("Leg 2: May 2025 - Sep 2026")
fig.tight_layout(); fig.savefig(RESULTS / "figures" / "bridge_calibration.png", dpi=200); plt.close(fig)

# %% [markdown]
# ## 3. Combining stations in the backtest: Clarity fills gaps in the PM2.5 inputs
# Per fold, Clarity is calibrated on hours before the training cut-off only; folds whose training data
# has fewer than 500 common hours (or r < 0.7) keep the original inputs. Original-input predictions are
# taken from Step 3 (same models, same test hours).

# %%
reg = pd.read_csv(ROOT / "data" / "predictions_regression.csv", parse_dates=["time"], index_col="time")
folds = make_folds(feat.index[feat.index >= CAMS_START], FIRST_TEST)
derived = [c for c in feat.columns if c.startswith("pm25_") and c != "pm25_obs"]
rows, dm_rows, fold_info = [], [], []
for h in HORIZONS:
    y, cols = feat[f"y_pm25_h{h}"], feature_list(h, True)
    preds = []
    for k, (a, b) in enumerate(folds):
        P = reg[(reg.h == h) & (reg.fold == k)][["y", "Ridge + CAMS", "LightGBM + CAMS"]].copy()
        if P.empty:
            continue
        cut = a - pd.Timedelta(hours=h)
        fit = fit_line(clarity[clarity.index < cut], feat["pm25_obs"][feat.index < cut])
        use = bool(fit and fit["r"] >= MIN_R)
        if use:
            filled = feat.copy()
            fill = (fit["intercept"] + fit["slope"] * clarity).clip(lower=1).reindex(feat.index)
            filled["pm25"] = filled["pm25_obs"].fillna(fill).interpolate(limit=3, limit_area="inside")
            filled = add_targets(add_features(filled.drop(columns=derived)))
            tr, _ = split_masks(feat.index, a, b, h, train_start=CAMS_START)
            tr &= y.notna()
            for name, make in [("Ridge + CAMS", ridge_regressor), ("LightGBM + CAMS", lgbm_regressor)]:
                P[name + " + Clarity gap-fill"] = np.maximum(
                    make().fit(filled.loc[tr, cols], y[tr]).predict(filled.loc[P.index, cols]), 0)
            upto = feat.index < b                       # inputs available up to the end of this test block
            extra = int(filled.loc[upto, "pm25"].notna().sum() - feat.loc[upto, "pm25"].notna().sum())
        else:
            for name in ["Ridge + CAMS", "LightGBM + CAMS"]:
                P[name + " + Clarity gap-fill"] = P[name]
            extra = 0
        fold_info.append({"h": h, "fold": k, "test_start": str(a.date()), "clarity_used": use,
                          "calibration_pairs": fit["pairs"] if fit else 0, "r": round(fit["r"], 3) if fit else None,
                          "input_hours_filled": extra})
        preds.append(P)
    P = pd.concat(preds).sort_index()
    for name in ["Ridge + CAMS", "LightGBM + CAMS"]:
        for variant in [name, name + " + Clarity gap-fill"]:
            rows.append({"h": h, "model": variant, **regression_metrics(P["y"], P[variant])})
        stat, pv = diebold_mariano(P[name + " + Clarity gap-fill"] - P["y"], P[name] - P["y"], h)
        dm_rows.append({"h": h, "model": name, "DM_gapfill_vs_original": stat, "p_value": pv})
    print(f"h={h} done")

ms, dm, fi = pd.DataFrame(rows), pd.DataFrame(dm_rows), pd.DataFrame(fold_info)
ms.round(3).to_csv(RESULTS / "multistation_backtest.csv", index=False)
dm.round(4).to_csv(RESULTS / "multistation_dm.csv", index=False)
fi.to_csv(RESULTS / "multistation_folds.csv", index=False)
pd.set_option("display.width", 200)
print("\nFolds where Clarity could be calibrated on training data:")
print(fi[fi.h == 24].to_string(index=False))
print("\nBacktest, original vs Clarity-gap-filled inputs (same test hours):")
print(ms.round(2).to_string(index=False))
print("\nDiebold-Mariano (negative = gap-filled better):")
print(dm.round(3).to_string(index=False))
