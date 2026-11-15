# %% [markdown]
# # Step 5b: Live forecasting model (weather + CAMS from Open-Meteo, no measured PM2.5)
# The US Embassy monitor's public feed ended in March 2025, so a live forecast cannot use lagged
# PM2.5 measurements. This model uses only inputs Open-Meteo provides in real time: current weather,
# the current CAMS value, and the CAMS forecast for the target hour. It is a learned correction of
# CAMS for Lahore.
#
# 1. Backtest it on the same 10 folds and the same test hours as Step 3 (honest live skill).
# 2. Train the final live models on all data and export them (models/meta_live.json).
# 3. Try one real live forecast for Lahore from Open-Meteo.
#
# Run after 06:   %run notebooks/07_live_model.py   (about 8-12 min on CPU)

# %%
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from src.config import CAMS_START, EVENT_THRESHOLD, HORIZONS, RESULTS, ROOT
from src.deploy import MODEL_DIR, Forecaster, aqi_category, ridge_to_dict
from src.evaluation import (best_threshold, blocked_oof, classification_metrics, diebold_mariano,
                            make_folds, regression_metrics, split_masks)
from src.features import add_live_features, add_targets, feature_list_live
from src.models import SmoteLGBM, lgbm_regressor, ridge_regressor

QUICK = os.getenv("PM25_QUICK") == "1"
FIRST_TEST = pd.Timestamp("2023-08-01")
COST = 10
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
t_start = time.time()

df = add_live_features(add_targets(pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"],
                                               index_col="time")))
folds = make_folds(df.index[df.index >= CAMS_START], FIRST_TEST)
if QUICK:
    folds = folds[:3]
reg = pd.read_csv(ROOT / "data" / "predictions_regression.csv", parse_dates=["time"], index_col="time")
cls = pd.read_csv(ROOT / "data" / "predictions_classification.csv", parse_dates=["time"], index_col="time")
cls = cls[cls.event.str.startswith("Very Unhealthy")]
LIVE_REG = {"Live Ridge (weather + CAMS)": ridge_regressor,
            "Live LightGBM (weather + CAMS)": lgbm_regressor}

# %% [markdown]
# ## 1. Backtest on the Step 3 folds and test hours

# %%
reg_out, cls_out = [], []
for h in HORIZONS:
    y, ye, cols = df[f"y_pm25_h{h}"], df[f"y_event_h{h}"], feature_list_live(h)
    for k, (a, b) in enumerate(folds):
        tr, _ = split_masks(df.index, a, b, h, train_start=CAMS_START)
        r = reg[(reg.h == h) & (reg.fold == k)][["h", "fold", "y", "Persistence", "Raw CAMS", "Ridge + CAMS"]].copy()
        if len(r):
            for name, make in LIVE_REG.items():
                m = make().fit(df.loc[tr & y.notna(), cols], y[tr & y.notna()])
                r[name] = np.maximum(m.predict(df.loc[r.index, cols]), 0)
            reg_out.append(r)

        c = cls[(cls.h == h) & (cls.fold == k)][["h", "fold", "y", "score|Raw CAMS", f"pred{COST}|Raw CAMS",
                                                  "score|LightGBM SMOTE", f"pred{COST}|LightGBM SMOTE"]].copy()
        X, yy = df.loc[tr & ye.notna(), cols], ye[tr & ye.notna()].astype(int)
        if len(c) and yy.sum() >= 6:
            p_oof = blocked_oof(SmoteLGBM, X, yy, h)
            ok = ~np.isnan(p_oof)
            thr = best_threshold(yy[ok].to_numpy(), p_oof[ok], fn_cost=COST)
            prob = SmoteLGBM().fit(X, yy).predict_proba(df.loc[c.index, cols])[:, 1]
            c["score|Live LightGBM SMOTE"], c[f"pred{COST}|Live LightGBM SMOTE"] = prob, (prob >= thr).astype(int)
            cls_out.append(c)
        print(f"h={h} fold {k} ({a:%Y-%m}) done [{(time.time() - t_start) / 60:.1f} min]")

R, C = pd.concat(reg_out), pd.concat(cls_out)

# %%
REG_MODELS = ["Persistence", "Raw CAMS"] + list(LIVE_REG) + ["Ridge + CAMS"]
rows, dm_rows = [], []
for h, g in R.groupby("h"):
    g = g.sort_index()
    base = {m: np.mean(np.abs(g[m] - g["y"])) for m in ["Persistence", "Raw CAMS"]}
    for m in REG_MODELS:
        met = regression_metrics(g["y"], g[m])
        rows.append({"h": h, "model": m, **met, "skill_vs_persistence": 1 - met["MAE"] / base["Persistence"],
                     "skill_vs_cams": 1 - met["MAE"] / base["Raw CAMS"]})
    for live in LIVE_REG:
        for other in ["Raw CAMS", "Persistence", "Ridge + CAMS"]:
            stat, p = diebold_mariano(g[live] - g["y"], g[other] - g["y"], int(h))
            dm_rows.append({"h": h, "live_model": live, "vs": other, "DM_stat": stat, "p_value": p})
live_reg, live_dm = pd.DataFrame(rows), pd.DataFrame(dm_rows)

ev_rows = []
for h, g in C.groupby("h"):
    for m in ["Raw CAMS", "Live LightGBM SMOTE", "LightGBM SMOTE"]:
        ev_rows.append({"h": h, "model": m if m != "LightGBM SMOTE" else "Full LightGBM SMOTE (measured PM2.5)",
                        **classification_metrics(g["y"], g[f"score|{m}"], fn_cost=COST, pred=g[f"pred{COST}|{m}"])})
live_ev = pd.DataFrame(ev_rows)

live_reg.round(3).to_csv(RESULTS / "live_model_regression.csv", index=False)
live_dm.round(4).to_csv(RESULTS / "live_model_dm_tests.csv", index=False)
live_ev.round(4).to_csv(RESULTS / "live_model_events.csv", index=False)
pd.set_option("display.width", 200)
for h in HORIZONS:
    print(f"\n=== Live model, h = {h} (same test hours as Step 3) ===")
    print(live_reg[live_reg.h == h].drop(columns="h").sort_values("MAE").round(2).to_string(index=False))
print("\nDiebold-Mariano (negative = live model better)")
print(live_dm.round(3).to_string(index=False))
print(f"\nSmog warnings (Very Unhealthy, cost {COST}:1)")
print(live_ev[["h", "model", "recall", "precision", "F1", "PR_AUC", "PR_AUC_random",
               "false_alarm_hours_per_30d", "cost"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.8))
show = ["Raw CAMS", "Persistence", min(LIVE_REG, key=lambda m: live_reg[live_reg.model == m].MAE.mean()),
        "Ridge + CAMS"]
colours = ["#eb6834", "#1baf7a", "#2a78d6", "#6250d6"]
width = 0.2
for i, (m, col) in enumerate(zip(show, colours)):
    s = live_reg[live_reg.model == m].sort_values("h")
    ax.bar(np.arange(len(HORIZONS)) + (i - 1.5) * width, s["MAE"], width * 0.9, color=col,
           label=m if m != "Ridge + CAMS" else "Full model (Ridge + CAMS, measured PM2.5)")
ax.set_xticks(range(len(HORIZONS)), [f"{h} h" for h in HORIZONS]); ax.set_ylabel("MAE (µg/m³)")
ax.set_title("Live model (no measured PM2.5) vs baselines and full model"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "live_model_backtest.png", dpi=200); plt.close(fig)

# %% [markdown]
# ## 2. Train the final live models on all data and export them

# %%
best = min(LIVE_REG, key=lambda m: live_reg[live_reg.model == m].MAE.mean())
print(f"\nDeployed live PM2.5 model: {best}")
short = df[df.index >= CAMS_START]
meta = {"name": "Live model (Open-Meteo weather + CAMS, no measured PM2.5)",
        "trained_on": [str(short.index.min()), str(short.index.max())],
        "regression_model": best, "event_model": "LightGBM + SMOTE (weather + CAMS)",
        "event_definition": f"24-h mean PM2.5 > {EVENT_THRESHOLD} µg/m³ at the target hour",
        "cost_ratio_missed_event_to_false_alarm": COST,
        "backtest_mae": {str(h): round(float(live_reg[(live_reg.model == best) & (live_reg.h == h)].MAE.iloc[0]), 1)
                         for h in HORIZONS},
        "pm25": {}, "event": {}}
fitted = {}
for h in HORIZONS:
    cols = feature_list_live(h)
    y, ye = short[f"y_pm25_h{h}"], short[f"y_event_h{h}"]
    model = LIVE_REG[best]().fit(short.loc[y.notna(), cols], y[y.notna()])
    if "Ridge" in best:
        (MODEL_DIR / f"live_pm25_h{h}.json").write_text(json.dumps(ridge_to_dict(model, cols)))
        meta["pm25"][str(h)] = {"format": "ridge_json", "file": f"live_pm25_h{h}.json", "features": cols}
    else:
        model.booster_.save_model(str(MODEL_DIR / f"live_pm25_h{h}.txt"))
        meta["pm25"][str(h)] = {"format": "lgbm_txt", "file": f"live_pm25_h{h}.txt", "features": cols}
    X, yy = short.loc[ye.notna(), cols], ye[ye.notna()].astype(int)
    p = blocked_oof(SmoteLGBM, X, yy, h)
    thr = best_threshold(yy[~np.isnan(p)].to_numpy(), p[~np.isnan(p)], fn_cost=COST)
    clf = SmoteLGBM().fit(X, yy)
    clf.model.booster_.save_model(str(MODEL_DIR / f"live_event_h{h}.txt"))
    meta["event"][str(h)] = {"file": f"live_event_h{h}.txt", "features": cols,
                             "median": clf.imp.statistics_.tolist(), "threshold": thr}
    fitted[h] = (model, clf, cols)
    print(f"h={h}: exported, warning threshold {thr:.3f}")
(MODEL_DIR / "meta_live.json").write_text(json.dumps(meta, indent=2))

fc = Forecaster("meta_live.json")
check = short.sample(500, random_state=0)
out = fc.predict(check)
for h, (model, clf, cols) in fitted.items():
    assert np.allclose(out[f"pm25_h{h}"], np.maximum(model.predict(check[cols]), 0), atol=1e-4)
    assert np.allclose(out[f"event_prob_h{h}"], clf.predict_proba(check[cols])[:, 1], atol=1e-6)
print("Exported live models reproduce the fitted models exactly.")

# %% [markdown]
# ## 3. A real live forecast for Lahore (Open-Meteo, right now)

# %%
try:
    from src.live import build_live
    res = build_live(api_key=None)            # weather + CAMS only; notebook 08 adds the live sensors
    row, t = res["row"], res["issue"]
    p = fc.predict(row).iloc[0]
    print(f"\nLive forecast issued {t:%Y-%m-%d %H:%M} PKT"
          + (f" (Open-Meteo did not provide: {res['missing_weather']})" if res["missing_weather"] else ""))
    for h in HORIZONS:
        pm = p[f"pm25_h{h}"]
        print(f"  +{h} h ({t + pd.Timedelta(hours=h):%d %b %H:%M}): {pm:6.1f} µg/m³ ({aqi_category(pm)}) | "
              f"raw CAMS {row[f'cams_h{h}'].iloc[0]:6.1f} | smog probability {p[f'event_prob_h{h}']:.2f} "
              f"{'-> WARNING' if p[f'event_warning_h{h}'] else ''}")
except Exception as e:                      # the export above is already saved either way
    print(f"\nLive test skipped (run notebook 08 first for the live configuration): {e}")

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")
