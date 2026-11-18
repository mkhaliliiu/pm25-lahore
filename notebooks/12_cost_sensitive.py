# %% [markdown]
# # Step 6: Cost-sensitive regression for high-intensity smog (sample weighting)
# Plain MAE/MSE training treats an error at 50 µg/m³ the same as an error at 600 µg/m³, and most hours
# are clean, so the models drift towards the mean and under-predict peaks. Here the training hours are
# weighted by intensity (and optionally by smog season), so errors on high-PM2.5 hours cost more.
# An asymmetric (quantile) LightGBM, which penalises under-prediction more than over-prediction, is
# tested alongside.
#
# Honest set-up:
#   * weights use only the training targets of each fold; test hours are never weighted;
#   * the weighting strength is chosen on data before the first test fold only (Jan-Jul 2023 inner
#     blocks, h = 48, as for Optuna), with a rule fixed in advance: lowest MAE on hours above 125.5
#     among settings whose overall MAE is at most 5% worse than no weighting;
#   * scored on exactly the same test hours as Step 3, against the unweighted models from Step 3.
#
# Run after 03:   %run notebooks/12_cost_sensitive.py   (about 10-15 min on CPU)
# Outputs: results/cost_sensitive_tuning.csv, cost_sensitive_summary.csv, cost_sensitive_by_band.csv,
#   cost_sensitive_dm.csv, cost_sensitive_bootstrap.csv, figures/cost_sensitive_bands.png,
#   figures/cost_sensitive_example.png; data/predictions_cost_sensitive.csv (large, gitignored)

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
from src.config import CAMS_START, HORIZONS, RESULTS, ROOT
from src.evaluation import (block_bootstrap_diff, diebold_mariano, make_folds, regression_metrics,
                            split_masks)
from src.features import add_targets, feature_list
from src.models import lgbm_regressor, ridge_regressor

FIRST_TEST = pd.Timestamp("2023-08-01")
VERY_UNHEALTHY, HAZARDOUS = 125.5, 225.5
H_TUNE = 48
MAX_OVERALL_LOSS = 1.05            # a setting may cost at most 5% overall MAE to be chosen
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
t_start = time.time()

df = add_targets(pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"], index_col="time"))
reg = pd.read_csv(ROOT / "data" / "predictions_regression.csv", parse_dates=["time"], index_col="time")
smog = lambda idx: idx.month.isin([10, 11, 12, 1, 2])


def sample_weights(y, target_index, alpha, season):
    """w = (y / median)^alpha, clipped to [0.2, 10], times `season` for Oct-Feb target hours,
    then scaled to mean 1. alpha = 0 and season = 1 means no weighting."""
    y = np.asarray(y, float)
    w = np.clip((np.maximum(y, 1) / np.median(y)) ** alpha, 0.2, 10)
    w = w * np.where(smog(target_index), season, 1.0)
    return w / w.mean()


def fit_predict(kind, params, Xtr, ytr, ttr, Xte):
    """kind: 'ridge' or 'lgbm' with sample weights, or 'quantile' (asymmetric loss, no weights)."""
    if kind == "quantile":
        model = lgbm_regressor({"objective": "quantile", "alpha": params["q"]}).fit(Xtr, ytr)
    else:
        w = sample_weights(ytr, ttr, params["alpha"], params["season"])
        if kind == "ridge":
            model = ridge_regressor().fit(Xtr, ytr, ridge__sample_weight=w)
        else:
            model = lgbm_regressor().fit(Xtr, ytr, sample_weight=w)
    return np.maximum(model.predict(Xte), 0)


def band_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"MAE_all": np.mean(np.abs(p - y))}
    for name, lo, hi in [("clean_<125.5", -np.inf, VERY_UNHEALTHY), ("very_unhealthy_125.5-225.5", VERY_UNHEALTHY, HAZARDOUS),
                         ("hazardous_>225.5", HAZARDOUS, np.inf)]:
        m = (y >= lo) & (y < hi)
        out[f"MAE_{name}"] = np.mean(np.abs(p[m] - y[m])) if m.any() else np.nan
        out[f"bias_{name}"] = np.mean(p[m] - y[m]) if m.any() else np.nan
        out[f"hours_{name}"] = int(m.sum())
    high = y >= VERY_UNHEALTHY
    out["MAE_high_>=125.5"] = np.mean(np.abs(p[high] - y[high])) if high.any() else np.nan
    return out


# %% [markdown]
# ## 1. Choose the weighting strength on data before the first test fold (h = 48)

# %%
inner = [(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-03-01")),
         (pd.Timestamp("2023-03-01"), pd.Timestamp("2023-05-01")),
         (pd.Timestamp("2023-05-01"), FIRST_TEST - pd.Timedelta(hours=H_TUNE))]
GRID = {"ridge": [{"alpha": a, "season": s} for a in (0.0, 0.5, 1.0, 1.5) for s in (1.0, 2.0)],
        "lgbm": [{"alpha": a, "season": s} for a in (0.0, 0.5, 1.0, 1.5) for s in (1.0, 2.0)],
        "quantile": [{"q": q} for q in (0.5, 0.6, 0.7, 0.8)]}
y_t, cols_t = df[f"y_pm25_h{H_TUNE}"], feature_list(H_TUNE, True)

tune_rows = []
for kind, grid in GRID.items():
    for params in grid:
        ys, ps = [], []
        for a, b in inner:
            tr, va = split_masks(df.index, a, b, H_TUNE, train_start=CAMS_START)
            tr &= y_t.notna()
            va &= y_t.notna()
            if tr.sum() < 500 or va.sum() < 100:
                continue
            ttr = df.index[tr] + pd.Timedelta(hours=H_TUNE)
            ps.append(fit_predict(kind, params, df.loc[tr, cols_t], y_t[tr], ttr, df.loc[va, cols_t]))
            ys.append(y_t[va].to_numpy())
        m = band_metrics(np.concatenate(ys), np.concatenate(ps))
        tune_rows.append({"kind": kind, **params, "inner_MAE_all": m["MAE_all"],
                          "inner_MAE_high": m["MAE_high_>=125.5"], "inner_hours_high": int((np.concatenate(ys) >= VERY_UNHEALTHY).sum())})
    print(f"tuned {kind} [{(time.time() - t_start) / 60:.1f} min]")
tune = pd.DataFrame(tune_rows)

chosen = {}
for kind in GRID:
    t = tune[tune.kind == kind]
    ref_kind = "ridge" if kind == "ridge" else "lgbm"       # quantile is judged against the plain LightGBM
    r = tune[(tune.kind == ref_kind) & (tune.alpha == 0) & (tune.season == 1)]
    ref = r.iloc[0]["inner_MAE_all"]
    ok = t[t.inner_MAE_all <= MAX_OVERALL_LOSS * ref]
    best = (ok.sort_values("inner_MAE_high") if len(ok) else t.sort_values("inner_MAE_all")).iloc[0]
    chosen[kind] = ({"alpha": float(best.alpha), "season": float(best.season)} if kind != "quantile"
                    else {"q": float(best.q)})
tune.round(3).to_csv(RESULTS / "cost_sensitive_tuning.csv", index=False)
pd.set_option("display.width", 220)
print(tune.round(2).to_string(index=False))
print("\nChosen settings (rule: lowest MAE on hours >= 125.5 with overall MAE <= 105% of unweighted):")
print(json.dumps(chosen, indent=2))
print("Also evaluated in every case: strong weighting alpha = 1, smog season x 2")

# %% [markdown]
# ## 2. Backtest on the Step 3 folds and test hours

# %%
FIXED = {"alpha": 1.0, "season": 2.0}   # a deliberately strong weighting, shown whatever the rule chose
VARIANTS = {"Ridge + CAMS (weighted)": ("ridge", chosen["ridge"]),
            "Ridge + CAMS (weighted, strong)": ("ridge", FIXED),
            "LightGBM + CAMS (weighted)": ("lgbm", chosen["lgbm"]),
            "LightGBM + CAMS (weighted, strong)": ("lgbm", FIXED),
            "LightGBM + CAMS (quantile loss)": ("quantile", chosen["quantile"])}
folds = make_folds(df.index[df.index >= CAMS_START], FIRST_TEST)
preds = []
for h in HORIZONS:
    y, cols = df[f"y_pm25_h{h}"], feature_list(h, True)
    for k, (a, b) in enumerate(folds):
        P = reg[(reg.h == h) & (reg.fold == k)][["h", "fold", "y", "Persistence", "Raw CAMS",
                                                  "Ridge + CAMS", "LightGBM + CAMS"]].copy()
        if P.empty:
            continue
        tr, _ = split_masks(df.index, a, b, h, train_start=CAMS_START)
        tr &= y.notna()
        assert not (df.index[tr] >= a - pd.Timedelta(hours=h)).any()          # no training target in the test block
        ttr = df.index[tr] + pd.Timedelta(hours=h)
        for name, (kind, params) in VARIANTS.items():
            P[name] = fit_predict(kind, params, df.loc[tr, cols], y[tr], ttr, df.loc[P.index, cols])
        preds.append(P)
    print(f"h={h} done [{(time.time() - t_start) / 60:.1f} min]")
pred = pd.concat(preds)
pred.to_csv(ROOT / "data" / "predictions_cost_sensitive.csv", index_label="time")

# %% [markdown]
# ## 3. Results: overall, by intensity band and season; DM tests; bootstrap on the high hours

# %%
MODELS = ["Persistence", "Raw CAMS", "Ridge + CAMS", "Ridge + CAMS (weighted)", "Ridge + CAMS (weighted, strong)",
          "LightGBM + CAMS", "LightGBM + CAMS (weighted)", "LightGBM + CAMS (weighted, strong)",
          "LightGBM + CAMS (quantile loss)"]
rows, band_rows, dm_rows, boot_rows = [], [], [], []
for h, g in pred.groupby("h"):
    g = g.sort_index()
    tgt = g.index + pd.Timedelta(hours=int(h))
    for m in MODELS:
        r = regression_metrics(g["y"], g[m])
        b = band_metrics(g["y"], g[m])
        r["MAE_smog_season"] = np.mean(np.abs(g[m] - g["y"])[smog(tgt)])
        rows.append({"h": h, "model": m, **{k: r[k] for k in ("MAE", "RMSE", "R2", "MAE_smog_season")},
                     "MAE_high_>=125.5": b["MAE_high_>=125.5"], "MAE_hazardous_>225.5": b["MAE_hazardous_>225.5"],
                     "bias_hazardous_>225.5": b["bias_hazardous_>225.5"], "MAE_clean_<125.5": b["MAE_clean_<125.5"]})
        band_rows.append({"h": h, "model": m, **b})
    for base, new in [("Ridge + CAMS", "Ridge + CAMS (weighted)"), ("Ridge + CAMS", "Ridge + CAMS (weighted, strong)"),
                      ("LightGBM + CAMS", "LightGBM + CAMS (weighted)"),
                      ("LightGBM + CAMS", "LightGBM + CAMS (weighted, strong)"),
                      ("LightGBM + CAMS", "LightGBM + CAMS (quantile loss)")]:
        if np.allclose(g[new], g[base], atol=1e-6):           # identical forecasts: no test to run
            stat, p = 0.0, 1.0
        else:
            stat, p = diebold_mariano(g[new] - g["y"], g[base] - g["y"], int(h))
        dm_rows.append({"h": h, "new": new, "vs": base, "DM_stat": stat, "p_value": p})
        high = g["y"] >= VERY_UNHEALTHY
        d, lo, hi = block_bootstrap_diff((g[new] - g["y"]).abs()[high], (g[base] - g["y"]).abs()[high], g.index[high])
        boot_rows.append({"h": h, "new": new, "vs": base, "hours": int(high.sum()),
                          "MAE_diff_high_hours": d, "ci95_low": lo, "ci95_high": hi})

summary, bands = pd.DataFrame(rows), pd.DataFrame(band_rows)
dm, boot = pd.DataFrame(dm_rows), pd.DataFrame(boot_rows)
summary.round(3).to_csv(RESULTS / "cost_sensitive_summary.csv", index=False)
bands.round(3).to_csv(RESULTS / "cost_sensitive_by_band.csv", index=False)
dm.round(4).to_csv(RESULTS / "cost_sensitive_dm.csv", index=False)
boot.round(3).to_csv(RESULTS / "cost_sensitive_bootstrap.csv", index=False)
(RESULTS / "cost_sensitive_settings.json").write_text(json.dumps({"chosen": chosen, "strong": FIXED}, indent=2))

for h in HORIZONS:
    print(f"\n=== h = {h} (same test hours as Step 3) ===")
    print(summary[summary.h == h].drop(columns="h").round(1).to_string(index=False))
print("\nDiebold-Mariano on all hours (negative = new model better):")
print(dm.round(3).to_string(index=False))
print("\nBootstrap 95% CI of the MAE difference on hours >= 125.5 (negative = new model better):")
print(boot.round(2).to_string(index=False))

# %% [markdown]
# ## 4. Figures

# %%
BANDS = [("MAE_clean_<125.5", "< 125.5"), ("MAE_very_unhealthy_125.5-225.5", "125.5–225.5"),
         ("MAE_hazardous_>225.5", "> 225.5")]
SHOW = ["Ridge + CAMS", "Ridge + CAMS (weighted, strong)", "LightGBM + CAMS", "LightGBM + CAMS (weighted, strong)",
        "LightGBM + CAMS (quantile loss)"]
COLOURS = ["#2a78d6", "#9ec5f4", "#eb6834", "#f6b89c", "#1baf7a"]      # fixed order per model
fig, axes = plt.subplots(1, len(HORIZONS), figsize=(12, 3.8), sharey=True)
for ax, h in zip(axes, HORIZONS):
    b = bands[bands.h == h].set_index("model")
    x = np.arange(len(BANDS))
    for i, (m, c) in enumerate(zip(SHOW, COLOURS)):
        ax.bar(x + (i - 2) * 0.16, [b.loc[m, col] for col, _ in BANDS], 0.15, color=c, label=m)
    ax.set_xticks(x, [lab for _, lab in BANDS]); ax.set_title(f"{h} h ahead"); ax.set_xlabel("Observed PM2.5 band (µg/m³)")
axes[0].set_ylabel("MAE (µg/m³)")
axes[-1].legend(fontsize=7, loc="upper left")
fig.suptitle("Error by intensity band: unweighted vs cost-sensitive training")
fig.tight_layout(); fig.savefig(FIG / "cost_sensitive_bands.png", dpi=200); plt.close(fig)

g = pred[pred.h == 24].copy()
g.index = g.index + pd.Timedelta(hours=24)
month = g["y"].resample("MS").mean().idxmax()
g = g[(g.index >= month) & (g.index < month + pd.DateOffset(months=1))]
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(g.index, g["y"], color="#52514e", lw=1.2, label="Observed")
ax.plot(g.index, g["Ridge + CAMS"], color="#2a78d6", lw=1, label="Ridge + CAMS")
ax.plot(g.index, g["Ridge + CAMS (weighted, strong)"], color="#c0392b", lw=1, label="Ridge + CAMS (weighted, strong)")
ax.axhline(VERY_UNHEALTHY, ls=":", color="#8a8985", lw=1)
ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_title(f"24-h-ahead forecasts, {month:%B %Y}: effect of weighting"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "cost_sensitive_example.png", dpi=200); plt.close(fig)
print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")
