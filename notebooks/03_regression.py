# %% [markdown]
# # Step 3a: PM2.5 forecasting at 24/48/72 h with a rolling-origin backtest (RQ1, RQ2)
# Run after 02_preprocess_eda.py:   %run notebooks/03_regression.py
# Takes roughly 15-25 minutes on a Colab CPU runtime (Random Forest is the slow part).
#
# Outputs (small, committed):  results/regression_summary.csv, regression_by_fold.csv,
#   regression_by_season.csv, dm_tests.csv, bootstrap_ci.csv, rq2_random_vs_rolling.csv,
#   optuna_best_params.json, figures/mae_by_horizon.png, mae_by_fold.png, example_forecast.png
# Output (large, gitignored):  data/predictions_regression.csv

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
import optuna
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path.cwd()))
from src.config import CAMS_START, HORIZONS, RESULTS, ROOT
from src.evaluation import (block_bootstrap_diff, diebold_mariano, make_folds,
                            regression_metrics, split_masks)
from src.features import add_targets, feature_list
from src.models import SEED, lgbm_regressor, rf_regressor, ridge_regressor, xgb_regressor

QUICK = os.getenv("PM25_QUICK") == "1"     # smoke test only: 3 folds, 3 Optuna trials
FIRST_TEST = pd.Timestamp("2023-08-01")    # first rolling-origin test block (2-month blocks after it)
N_TRIALS = 3 if QUICK else 25
REF = "LightGBM + CAMS (tuned)"            # main model, compared against everything else
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
t_start = time.time()

df = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"], index_col="time")
df = add_targets(df)                       # recomputes targets and CAMS columns (cheap)
for h in HORIZONS:
    df[f"naive7_h{h}"] = df["pm25"].shift(168 - h)   # same hour one week before the target
smog = lambda idx: idx.month.isin([10, 11, 12, 1, 2])
print(f"Rows {len(df)} | {df.index.min()} to {df.index.max()} | quick={QUICK}")

# %% [markdown]
# ## 1. Rolling-origin folds
# Expanding training window, 2-month test blocks, and a gap of h hours so that no training
# target falls inside the test block.

# %%
folds = make_folds(df.index[df.index >= CAMS_START], FIRST_TEST)
if QUICK:
    folds = folds[:3]
for k, (a, b) in enumerate(folds):
    print(f"fold {k}: test {a:%Y-%m-%d} to {b:%Y-%m-%d}")

# %% [markdown]
# ## 2. Optuna tuning of LightGBM + CAMS (h = 48), using only data before the first test fold
# Inner validation blocks run Jan-Jul 2023, so the test period is never seen during tuning.

# %%
H_TUNE = 48
inner = [(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-03-01")),
         (pd.Timestamp("2023-03-01"), pd.Timestamp("2023-05-01")),
         (pd.Timestamp("2023-05-01"), FIRST_TEST - pd.Timedelta(hours=H_TUNE))]
y_tune = df[f"y_pm25_h{H_TUNE}"]
cols_tune = feature_list(H_TUNE, True)


def inner_mae(params):
    maes = []
    for a, b in inner:
        tr, va = split_masks(df.index, a, b, H_TUNE, train_start=CAMS_START)
        tr &= y_tune.notna()
        va &= y_tune.notna()
        if tr.sum() < 500 or va.sum() < 100:
            continue
        m = lgbm_regressor(params).fit(df.loc[tr, cols_tune], y_tune[tr])
        maes.append(np.mean(np.abs(m.predict(df.loc[va, cols_tune]) - y_tune[va])))
    return float(np.mean(maes))


def objective(trial):
    return inner_mae(dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        log_target=trial.suggest_categorical("log_target", [False, True]),
    ))


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.enqueue_trial(dict(num_leaves=31, learning_rate=0.03, n_estimators=600, min_child_samples=30,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, log_target=False))
study.optimize(objective, n_trials=N_TRIALS)
BEST = study.best_params
tuning = {"horizon": H_TUNE, "trials": N_TRIALS, "inner_blocks": [[str(a), str(b)] for a, b in inner],
          "default_inner_mae": round(study.trials[0].value, 2),
          "best_inner_mae": round(study.best_value, 2), "best_params": BEST}
with open(RESULTS / "optuna_best_params.json", "w") as f:
    json.dump(tuning, f, indent=2)
print(json.dumps(tuning, indent=2))

# %% [markdown]
# ## 3. Backtest: baselines and models, every fold and horizon
# All models are scored on the same test hours: target observed, current PM2.5 known,
# CAMS available, and the weekly-naive value available.

# %%
MODELS = {   # name: (factory, uses CAMS, training window)
    "Ridge + CAMS": (ridge_regressor, True, "short"),
    "Random Forest + CAMS": (rf_regressor, True, "short"),
    "XGBoost + CAMS": (xgb_regressor, True, "short"),
    "LightGBM (no CAMS)": (lgbm_regressor, False, "short"),
    "LightGBM long history (no CAMS)": (lgbm_regressor, False, "long"),
    "LightGBM + CAMS": (lgbm_regressor, True, "short"),
    REF: (lambda: lgbm_regressor(BEST), True, "short"),
}
BASELINES = ["Persistence", "Weekly naive", "Climatology", "Raw CAMS"]
ALL = BASELINES + list(MODELS)


def climatology(train_idx, y_train, test_idx, h):
    """Mean observed PM2.5 by (month, hour) of the target time, from training data only."""
    tt = train_idx + pd.Timedelta(hours=h)
    table = y_train.groupby([tt.month, tt.hour]).mean()
    qt = test_idx + pd.Timedelta(hours=h)
    keys = pd.MultiIndex.from_arrays([qt.month, qt.hour])
    return table.reindex(keys).fillna(y_train.mean()).to_numpy()


preds = []
for h in HORIZONS:
    y = df[f"y_pm25_h{h}"]
    scored = y.notna() & df["pm25"].notna() & df[f"cams_h{h}"].notna() & df[f"naive7_h{h}"].notna()
    for k, (a, b) in enumerate(folds):
        _, te = split_masks(df.index, a, b, h)
        te &= scored
        if te.sum() < 100:
            print(f"h={h} fold {k}: only {te.sum()} scored hours, skipped")
            continue
        out = pd.DataFrame(index=df.index[te])
        out["h"], out["fold"], out["y"] = h, k, y[te]
        out["Persistence"] = df.loc[te, "pm25"]
        out["Weekly naive"] = df.loc[te, f"naive7_h{h}"]
        out["Raw CAMS"] = df.loc[te, f"cams_h{h}"]
        tr_s, _ = split_masks(df.index, a, b, h, train_start=CAMS_START)
        tr_s &= y.notna()
        out["Climatology"] = climatology(df.index[tr_s], y[tr_s], df.index[te], h)
        for name, (make, cams, window) in MODELS.items():
            tr, _ = split_masks(df.index, a, b, h, train_start=CAMS_START if window == "short" else None)
            tr &= y.notna()
            cols = feature_list(h, cams)
            model = make().fit(df.loc[tr, cols], y[tr])
            out[name] = np.maximum(model.predict(df.loc[te, cols]), 0)
        preds.append(out)
        print(f"h={h} fold {k} ({a:%Y-%m}): n={te.sum():5d}  "
              f"MAE persistence {np.mean(np.abs(out['Persistence'] - out['y'])):6.1f}  "
              f"CAMS {np.mean(np.abs(out['Raw CAMS'] - out['y'])):6.1f}  "
              f"{REF} {np.mean(np.abs(out[REF] - out['y'])):6.1f}  "
              f"[{(time.time() - t_start) / 60:.1f} min]")

pred = pd.concat(preds)
pred.to_csv(ROOT / "data" / "predictions_regression.csv", index_label="time")

# %% [markdown]
# ## 4. Results: pooled metrics, fold spread, skill scores, seasons

# %%
rows, fold_rows, season_rows = [], [], []
for h, g in pred.groupby("h"):
    base = {m: np.mean(np.abs(g[m] - g["y"])) for m in ["Persistence", "Raw CAMS"]}
    for m in ALL:
        r = regression_metrics(g["y"], g[m])
        per_fold = (g[m] - g["y"]).abs().groupby(g["fold"]).mean()
        r.update(h=h, model=m, MAE_fold_mean=per_fold.mean(), MAE_fold_std=per_fold.std(),
                 skill_vs_persistence=1 - r["MAE"] / base["Persistence"],
                 skill_vs_cams=1 - r["MAE"] / base["Raw CAMS"])
        rows.append(r)
        for k, f in g.groupby("fold"):
            fold_rows.append({"h": h, "fold": k, "test_start": str(f.index.min().date()), "model": m,
                              **regression_metrics(f["y"], f[m])})
        for s, f in g.groupby(np.where(smog(g.index + pd.Timedelta(hours=h)), "Oct-Feb", "Mar-Sep")):
            season_rows.append({"h": h, "season": s, "model": m, **regression_metrics(f["y"], f[m])})

summary = pd.DataFrame(rows)[["h", "model", "n", "MAE", "RMSE", "R2", "MAPE", "MAE_fold_mean",
                              "MAE_fold_std", "skill_vs_persistence", "skill_vs_cams"]]
by_fold, by_season = pd.DataFrame(fold_rows), pd.DataFrame(season_rows)
summary.round(3).to_csv(RESULTS / "regression_summary.csv", index=False)
by_fold.round(3).to_csv(RESULTS / "regression_by_fold.csv", index=False)
by_season.round(3).to_csv(RESULTS / "regression_by_season.csv", index=False)

pd.set_option("display.width", 200)
for h in HORIZONS:
    print(f"\n=== Horizon {h} h (pooled over {pred.fold[pred.h == h].nunique()} folds) ===")
    print(summary[summary.h == h].drop(columns="h").sort_values("MAE").round(2).to_string(index=False))
print("\nMAE by season (Oct-Feb = smog season):")
print(by_season.pivot_table(index="model", columns=["h", "season"], values="MAE").round(1).to_string())

# %% [markdown]
# ## 5. Statistical tests: Diebold-Mariano (HLN) and day-block bootstrap CIs
# Negative DM statistic / negative difference = the tuned LightGBM + CAMS has lower error.

# %%
dm_rows, boot_rows = [], []
for h, g in pred.groupby("h"):
    g = g.sort_index()
    e_ref = g[REF] - g["y"]
    for m in ALL:
        if m == REF:
            continue
        e = g[m] - g["y"]
        stat, p = diebold_mariano(e_ref, e, h)
        dm_rows.append({"h": h, "vs": m, "MAE_ref": np.mean(np.abs(e_ref)), "MAE_other": np.mean(np.abs(e)),
                        "DM_stat": stat, "p_value": p})
        if m in ["Persistence", "Raw CAMS", "LightGBM (no CAMS)", "LightGBM long history (no CAMS)",
                 "LightGBM + CAMS"]:
            d, lo, hi = block_bootstrap_diff(np.abs(e_ref), np.abs(e), g.index)
            boot_rows.append({"h": h, "vs": m, "MAE_difference": d, "ci95_low": lo, "ci95_high": hi})

dm, boot = pd.DataFrame(dm_rows), pd.DataFrame(boot_rows)
dm.round(4).to_csv(RESULTS / "dm_tests.csv", index=False)
boot.round(3).to_csv(RESULTS / "bootstrap_ci.csv", index=False)
print(f"\nDiebold-Mariano: {REF} vs others")
print(dm.round(3).to_string(index=False))
print("\nBootstrap 95% CI of MAE difference (ref minus other; below 0 = ref better)")
print(boot.round(2).to_string(index=False))

# %% [markdown]
# ## 6. RQ2: how optimistic is a random split?
# Same model and features, but a shuffled 5-fold split of the short window (80/20 each time).

# %%
rq2 = []
for h in HORIZONS:
    y = df[f"y_pm25_h{h}"]
    rows_ok = (df.index >= CAMS_START) & y.notna() & df["pm25"].notna() & df[f"cams_h{h}"].notna()
    X, yy, cols = df.loc[rows_ok], y[rows_ok], feature_list(h, True)
    p = np.zeros(len(X))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        p[te] = np.maximum(lgbm_regressor(BEST).fit(X.iloc[tr][cols], yy.iloc[tr]).predict(X.iloc[te][cols]), 0)
    rand = regression_metrics(yy, p)
    roll = summary[(summary.h == h) & (summary.model == REF)].iloc[0]
    rq2.append({"h": h, "random_MAE": rand["MAE"], "random_R2": rand["R2"],
                "rolling_MAE": roll["MAE"], "rolling_R2": roll["R2"],
                "MAE_underestimated_by_pct": 100 * (roll["MAE"] - rand["MAE"]) / roll["MAE"]})
rq2 = pd.DataFrame(rq2)
rq2.round(3).to_csv(RESULTS / "rq2_random_vs_rolling.csv", index=False)
print("\nRQ2: random 80/20 split vs rolling-origin backtest")
print(rq2.round(2).to_string(index=False))

# %% [markdown]
# ## 7. Figures for the paper

# %%
show = ["Persistence", "Climatology", "Raw CAMS", "Ridge + CAMS", "Random Forest + CAMS",
        "XGBoost + CAMS", "LightGBM (no CAMS)", "LightGBM long history (no CAMS)", REF]
fig, ax = plt.subplots(figsize=(7, 4))
for m in show:
    s = summary[summary.model == m].sort_values("h")
    ax.plot(s["h"], s["MAE"], marker="o", lw=2.5 if m == REF else 1.2, label=m)
ax.set_xticks(list(HORIZONS)); ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("MAE (µg/m³)")
ax.set_title("Rolling-origin MAE by horizon"); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "mae_by_horizon.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.5))
for m in ["Persistence", "Raw CAMS", REF]:
    s = by_fold[(by_fold.model == m) & (by_fold.h == 24)]
    ax.plot(pd.to_datetime(s["test_start"]), s["MAE"], marker="o", label=m)
ax.set_ylabel("MAE (µg/m³)"); ax.set_title("MAE per 2-month test fold, h = 24"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "mae_by_fold.png", dpi=200); plt.close(fig)

g = pred[pred.h == 24].copy()
g.index = g.index + pd.Timedelta(hours=24)                # plot against the target time
month = g["y"].resample("MS").mean().idxmax()             # the worst smog month in the test period
g = g[(g.index >= month) & (g.index < month + pd.DateOffset(months=1))]
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(g.index, g["y"], color="black", lw=1.2, label="Observed")
ax.plot(g.index, g[REF], color="#c0392b", lw=1, label=REF)
ax.plot(g.index, g["Raw CAMS"], color="#1f4e79", lw=1, ls="--", label="Raw CAMS")
ax.plot(g.index, g["Persistence"], color="grey", lw=0.8, alpha=0.7, label="Persistence")
ax.axhline(125.5, ls=":", color="#c0392b", lw=1)
ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_title(f"24-h-ahead forecasts, {month:%B %Y}"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "example_forecast.png", dpi=200); plt.close(fig)

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Figures:", sorted(p.name for p in FIG.glob("*.png")))
