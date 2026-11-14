# %% [markdown]
# # Step 5a: Final models, data snapshots for the dashboard, and the cost curve
# Run after 03 and 04:   %run notebooks/06_export_models.py   (about 3-5 min on CPU)
#
# Chosen from the backtest:
#   PM2.5 forecast  -> Ridge + CAMS (lowest MAE at 48 and 72 h, within noise of the best at 24 h)
#   Smog warning    -> LightGBM + SMOTE (lowest cost at 10:1 on most horizons, best lead time)
# Final models are trained on all data with CAMS (Aug 2022 onward). Their accuracy is the
# backtest's, not this fit's.
#
# Outputs (committed, small):  models/ridge_h*.json, models/event_h*.txt, models/meta.json,
#   app/data/*.csv.gz, results/cost_curve.csv, results/figures/cost_curve.png

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
from src.config import CAMS_START, EVENT_THRESHOLD, HORIZONS, RESULTS, ROOT
from src.deploy import MODEL_DIR, Forecaster, ridge_to_dict
from src.evaluation import best_threshold, blocked_oof
from src.features import add_targets, feature_list
from src.models import SmoteLGBM, ridge_regressor

APP_DATA = ROOT / "app" / "data"
APP_DATA.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
COST = 10

df = add_targets(pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"], index_col="time"))
short = df[df.index >= CAMS_START]
print(f"Training window: {short.index.min()} to {short.index.max()} ({len(short)} hours)")

# %% [markdown]
# ## 1. Fit and export the final models

# %%
meta = {"name": "Full model (uses measured PM2.5)",
        "trained_on": [str(short.index.min()), str(short.index.max())],
        "station": "US Diplomatic Post Lahore (OpenAQ sensor 25135)",
        "regression_model": "Ridge + CAMS", "event_model": "LightGBM + SMOTE",
        "event_definition": f"observed 24-h mean PM2.5 > {EVENT_THRESHOLD} µg/m³ at the target hour",
        "cost_ratio_missed_event_to_false_alarm": COST, "pm25": {}, "event": {}}
fitted = {}
for h in HORIZONS:
    cols = feature_list(h, True)
    y = short[f"y_pm25_h{h}"]
    ok = y.notna()
    ridge = ridge_regressor().fit(short.loc[ok, cols], y[ok])
    (MODEL_DIR / f"ridge_h{h}.json").write_text(json.dumps(ridge_to_dict(ridge, cols)))
    meta["pm25"][str(h)] = {"format": "ridge_json", "file": f"ridge_h{h}.json", "features": cols}

    ye = short[f"y_event_h{h}"]
    X, yy = short.loc[ye.notna(), cols], ye[ye.notna()].astype(int)
    p = blocked_oof(SmoteLGBM, X, yy, h)
    thr = best_threshold(yy[~np.isnan(p)].to_numpy(), p[~np.isnan(p)], fn_cost=COST)
    clf = SmoteLGBM().fit(X, yy)
    clf.model.booster_.save_model(str(MODEL_DIR / f"event_h{h}.txt"))
    meta["event"][str(h)] = {"file": f"event_h{h}.txt", "features": cols, "median": clf.imp.statistics_.tolist(),
                             "threshold": thr, "training_hours": int(len(yy)), "event_rate": round(float(yy.mean()), 3)}
    fitted[h] = (ridge, clf, cols)
    print(f"h={h}: Ridge on {ok.sum()} hours | event model on {len(yy)} hours "
          f"(event rate {yy.mean():.2f}) | threshold {thr:.3f}")

(MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

# %% [markdown]
# ## 2. Check: the exported files give the same predictions as the fitted models

# %%
fc = Forecaster("meta.json")
check = short.sample(500, random_state=0)
out = fc.predict(check)
for h, (ridge, clf, cols) in fitted.items():
    assert np.allclose(out[f"pm25_h{h}"], np.maximum(ridge.predict(check[cols]), 0), atol=1e-6)
    assert np.allclose(out[f"event_prob_h{h}"], clf.predict_proba(check[cols])[:, 1], atol=1e-6)
print("Exported models reproduce the fitted models exactly.")

# %% [markdown]
# ## 3. Small data snapshots for the dashboard (all out-of-sample backtest results)

# %%
short[["pm25_obs", "cams_pm25"]].round(1).to_csv(APP_DATA / "observed.csv.gz", index_label="time")

reg = pd.read_csv(ROOT / "data" / "predictions_regression.csv", parse_dates=["time"])
reg = reg[["time", "h", "fold", "y", "Persistence", "Raw CAMS", "Ridge + CAMS",
           "Random Forest + CAMS", "LightGBM + CAMS"]]
cls_all = pd.read_csv(ROOT / "data" / "predictions_classification.csv", parse_dates=["time"])
cls = cls_all[cls_all.event.str.startswith("Very Unhealthy")][
    ["time", "h", "y", "score|LightGBM SMOTE", f"pred{COST}|LightGBM SMOTE"]].rename(columns={
        "y": "event_observed", "score|LightGBM SMOTE": "event_prob", f"pred{COST}|LightGBM SMOTE": "event_warning"})
bt = reg.merge(cls, on=["time", "h"], how="left")
bt.set_index("time").round(3).to_csv(APP_DATA / "backtest.csv.gz", index_label="time")

recent = short[short.index >= short.index.max() - pd.Timedelta(days=60)]
recent[["pm25_obs"] + fc.features].round(4).to_csv(APP_DATA / "recent_features.csv.gz", index_label="time")

# %% [markdown]
# ## 4. Threshold vs expected cost (proposal section 8.3)
# Pooled out-of-sample probabilities from Step 3b; cost = 10 x missed event hours + false-alarm hours.
# The dot on each curve is the mean threshold that the training-only procedure actually chose.

# %%
VARIANTS = ["LightGBM plain", "LightGBM class weights", "LightGBM SMOTE"]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]          # fixed order: one colour per variant
grid = np.unique(np.concatenate([np.geomspace(0.001, 0.1, 60), np.linspace(0.1, 0.99, 90)]))
ev = cls_all[cls_all.event.str.startswith("Very Unhealthy")]
rows = []
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(12, 3.6), sharey=False)
for ax, h in zip(axes, HORIZONS):
    g = ev[ev.h == h]
    y = g["y"].to_numpy()
    for v, colour in zip(VARIANTS, COLOURS):
        p = g[f"score|{v}"].to_numpy()
        fn = np.array([((p < t) & (y == 1)).sum() for t in grid])
        fp = np.array([((p >= t) & (y == 0)).sum() for t in grid])
        cost = COST * fn + fp
        rows += [{"h": h, "model": v, "threshold": t, "FN": a, "FP": b, "cost": c}
                 for t, a, b, c in zip(grid, fn, fp, cost)]
        ax.plot(grid, cost, color=colour, lw=2, label=v)
        chosen = g[f"thr{COST}|{v}"].mean()
        ax.plot(chosen, COST * ((p < chosen) & (y == 1)).sum() + ((p >= chosen) & (y == 0)).sum(),
                "o", color=colour, ms=8, mec="white", mew=2)
    ax.set_xscale("log"); ax.set_title(f"{h} h ahead"); ax.set_xlabel("Decision threshold (log scale)")
axes[0].set_ylabel("Expected cost (10·FN + FP)")
axes[0].legend(fontsize=8)
fig.suptitle("Threshold vs expected cost, Very Unhealthy event (dots = threshold chosen on training data)")
fig.tight_layout(); fig.savefig(RESULTS / "figures" / "cost_curve.png", dpi=200); plt.close(fig)
pd.DataFrame(rows).to_csv(RESULTS / "cost_curve.csv", index=False)
print("Saved results/figures/cost_curve.png")

for f in sorted(list(MODEL_DIR.glob("*")) + list(APP_DATA.glob("*"))):
    print(f"{f.relative_to(ROOT)}: {f.stat().st_size / 1024:.0f} KB")
