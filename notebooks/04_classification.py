# %% [markdown]
# # Step 3b: Severe-smog event prediction on imbalanced data (RQ3)
# Event = observed 24-h mean PM2.5 at the target time > 125.5 µg/m³ (US EPA "Very Unhealthy").
# Compares LightGBM plain vs class weights vs SMOTE, with a missed-event cost of 10x a false alarm
# (sensitivity at 5x and 20x), and a rarer-event check at 225.5 µg/m³ ("Hazardous").
#
# Run after 03_regression.py:   %run notebooks/04_classification.py   (about 15-25 min on CPU)
# Outputs: results/classification_summary.csv, classification_by_season.csv, lead_time.csv,
#          figures/pr_curves_h48.png, figures/recall_by_season.png; data/predictions_classification.csv

# %%
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path.cwd()))
from src.config import CAMS_START, EVENT_THRESHOLD, HORIZONS, RESULTS, ROOT
from src.evaluation import best_threshold, classification_metrics, make_folds, split_masks
from src.features import add_targets, feature_list
from src.models import SEED, lgbm_classifier

warnings.filterwarnings("ignore", category=UserWarning)
QUICK = os.getenv("PM25_QUICK") == "1"
FIRST_TEST = pd.Timestamp("2023-08-01")
COSTS = (5, 10, 20)                   # cost of a missed event relative to a false alarm
MAIN_COST = 10
EVENTS = {"Very Unhealthy (>125.5)": (EVENT_THRESHOLD, HORIZONS),
          "Hazardous (>225.5)": (225.5, (24,))}          # rarer-event sensitivity check at h = 24
VARIANTS = ["LightGBM plain", "LightGBM class weights", "LightGBM SMOTE"]
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
t_start = time.time()

df = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"], index_col="time")
df = add_targets(df)
folds = make_folds(df.index[df.index >= CAMS_START], FIRST_TEST)
if QUICK:
    folds = folds[:3]
smog = lambda idx: idx.month.isin([10, 11, 12, 1, 2])

# %% [markdown]
# ## 1. Helpers: the three imbalance strategies and an out-of-fold threshold
# The decision threshold is chosen on training data only: blocked 3-fold out-of-fold
# probabilities (contiguous time blocks, with a purge gap), minimising 10*FN + FP.


class SmoteLGBM:
    """Median-impute, oversample the event class with SMOTE (training data only), then LightGBM."""
    def fit(self, X, y):
        self.imp = SimpleImputer(strategy="median").fit(X)
        Xs, ys = SMOTE(random_state=SEED, k_neighbors=min(5, int(y.sum()) - 1)).fit_resample(
            self.imp.transform(X), y)
        self.model = lgbm_classifier().fit(Xs, ys)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.imp.transform(X))


class Constant:
    def __init__(self, p): self.p = p
    def predict_proba(self, X): return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def fit_variant(name, X, y):
    if y.nunique() < 2 or y.sum() < 6:
        return Constant(float(y.mean()))
    if name == "LightGBM plain":
        return lgbm_classifier().fit(X, y)
    if name == "LightGBM class weights":
        return lgbm_classifier(scale_pos_weight=float((y == 0).sum() / (y == 1).sum())).fit(X, y)
    return SmoteLGBM().fit(X, y)


def oof_probs(name, X, y, h, n_blocks=3):
    """Out-of-fold probabilities on the training rows, using contiguous time blocks."""
    p = np.full(len(X), np.nan)
    edges = np.linspace(0, len(X), n_blocks + 1).astype(int)
    purge = pd.Timedelta(hours=h + 48)          # target horizon plus the longest lag
    for i in range(n_blocks):
        lo, hi = X.index[edges[i]], X.index[edges[i + 1] - 1]
        val = (X.index >= lo) & (X.index <= hi)
        fit = (X.index < lo - purge) | (X.index > hi + purge)
        p[val] = fit_variant(name, X[fit], y[fit]).predict_proba(X[val])[:, 1]
    return p


# %% [markdown]
# ## 2. Backtest
# Baselines use a PM2.5 value as the score (for PR-AUC) and the event threshold as the cut-off:
# persistence = current 24-h mean; raw CAMS = CAMS forecast of the target-day mean.

# %%
preds = []
for ev_name, (thr, horizons) in EVENTS.items():
    for h in horizons:
        mean24_target = df["mean24_obs"].shift(-h)
        y = (mean24_target > thr).astype(float).where(mean24_target.notna())
        scored = y.notna() & df["pm25_mean24"].notna() & df[f"cams_mean24_h{h}"].notna()
        cols = feature_list(h, True)
        for k, (a, b) in enumerate(folds):
            tr, te = split_masks(df.index, a, b, h, train_start=CAMS_START)
            tr &= y.notna()
            te &= scored
            if te.sum() < 100:
                continue
            out = pd.DataFrame(index=df.index[te])
            out["event"], out["h"], out["fold"], out["y"] = ev_name, h, k, y[te].astype(int)
            out["score|Persistence"] = df.loc[te, "pm25_mean24"]
            out["score|Raw CAMS"] = df.loc[te, f"cams_mean24_h{h}"]
            for base in ["Persistence", "Raw CAMS"]:
                for c in COSTS:
                    out[f"pred{c}|{base}"] = (out[f"score|{base}"] > thr).astype(int)
            out["score|No-event (always 0)"] = 0.0
            for c in COSTS:
                out[f"pred{c}|No-event (always 0)"] = 0

            Xtr, ytr, Xte = df.loc[tr, cols], y[tr].astype(int), df.loc[te, cols]
            for v in VARIANTS:
                p_oof = oof_probs(v, Xtr, ytr, h)
                ok = ~np.isnan(p_oof)
                prob = fit_variant(v, Xtr, ytr).predict_proba(Xte)[:, 1]
                out[f"score|{v}"] = prob
                for c in COSTS:
                    t = best_threshold(ytr[ok].to_numpy(), p_oof[ok], fn_cost=c)
                    out[f"pred{c}|{v}"] = (prob >= t).astype(int)
                    out[f"thr{c}|{v}"] = t
            preds.append(out)
            print(f"{ev_name} h={h} fold {k} ({a:%Y-%m}): n={te.sum():5d} events={int(y[te].sum()):5d} "
                  f"[{(time.time() - t_start) / 60:.1f} min]")

pred = pd.concat(preds)
pred.to_csv(ROOT / "data" / "predictions_classification.csv", index_label="time")
MODELS = ["No-event (always 0)", "Persistence", "Raw CAMS"] + VARIANTS

# %% [markdown]
# ## 3. Pooled metrics at each cost ratio, and recall by season

# %%
rows, season_rows = [], []
for (ev, h), g in pred.groupby(["event", "h"]):
    tgt_smog = smog(g.index + pd.Timedelta(hours=int(h)))
    for m in MODELS:
        for c in COSTS:
            r = classification_metrics(g["y"], g[f"score|{m}"], fn_cost=c, pred=g[f"pred{c}|{m}"])
            if f"thr{c}|{m}" in g:
                r["threshold"] = g[f"thr{c}|{m}"].mean()
            rows.append({"event": ev, "h": h, "model": m, "cost_ratio": c, **r})
        for s, mask in [("Oct-Feb", tgt_smog), ("Mar-Sep", ~tgt_smog)]:
            f = g[mask]
            r = classification_metrics(f["y"], f[f"score|{m}"], fn_cost=MAIN_COST, pred=f[f"pred{MAIN_COST}|{m}"])
            season_rows.append({"event": ev, "h": h, "season": s, "model": m, **r})

summary, by_season = pd.DataFrame(rows), pd.DataFrame(season_rows)
summary.round(4).to_csv(RESULTS / "classification_summary.csv", index=False)
by_season.round(4).to_csv(RESULTS / "classification_by_season.csv", index=False)

show = ["model", "recall", "precision", "F1", "PR_AUC", "PR_AUC_random", "ROC_AUC", "accuracy",
        "false_alarms_per_30d", "FN", "FP", "cost", "threshold"]
pd.set_option("display.width", 220)
for (ev, h), s in summary[summary.cost_ratio == MAIN_COST].groupby(["event", "h"]):
    print(f"\n=== {ev}, h = {h}, cost ratio {MAIN_COST}:1 (n = {s.n.iloc[0]}, events = {s.positives.iloc[0]}) ===")
    print(s[show].round(3).to_string(index=False))

print("\nCost-ratio sensitivity (Very Unhealthy): recall / false alarms per 30 days")
sens = summary[(summary.event == list(EVENTS)[0]) & summary.model.isin(VARIANTS)]
print(sens.pivot_table(index=["h", "model"], columns="cost_ratio",
                       values=["recall", "false_alarms_per_30d"]).round(2).to_string())
print("\nRecall by season (cost 10:1):")
print(by_season.pivot_table(index=["event", "h", "model"], columns="season", values="recall").round(3).to_string())

# %% [markdown]
# ## 4. Lead time: how early is the onset of each smog episode flagged?
# Onset = an hour whose observed 24-h mean is above 125.5 after at least 24 hours below it
# (so short flickers around the threshold inside one episode are not counted as new episodes).
# Lead time = the longest horizon (24/48/72 h) whose forecast for that hour was "event".

# %%
ev0 = list(EVENTS)[0]
obs = (df["mean24_obs"] > EVENT_THRESHOLD).astype(float).where(df["mean24_obs"].notna())
quiet_before = obs.shift(1).rolling(24, min_periods=12).max() == 0
onsets = obs.index[(obs == 1) & quiet_before]
lead_rows = []
for m in MODELS[1:]:
    hit = {}
    for h in HORIZONS:
        g = pred[(pred.event == ev0) & (pred.h == h)]
        hit[h] = pd.Series(g[f"pred{MAIN_COST}|{m}"].to_numpy(), index=g.index + pd.Timedelta(hours=h))
    covered = [t for t in onsets if all(t in hit[h].index for h in HORIZONS)]
    leads = np.array([max([h for h in HORIZONS if hit[h][t] == 1], default=0) for t in covered])
    lead_rows.append({"model": m, "onsets_scored": len(covered),
                      "detected_pct": 100 * np.mean(leads > 0) if len(leads) else np.nan,
                      "detected_48h_or_more_pct": 100 * np.mean(leads >= 48) if len(leads) else np.nan,
                      "detected_72h_pct": 100 * np.mean(leads >= 72) if len(leads) else np.nan,
                      "median_lead_h_when_detected": np.median(leads[leads > 0]) if (leads > 0).any() else 0})
lead = pd.DataFrame(lead_rows)
lead.round(2).to_csv(RESULTS / "lead_time.csv", index=False)
print("\nLead time at episode onsets (cost 10:1):")
print(lead.round(1).to_string(index=False))

# %% [markdown]
# ## 5. Figures

# %%
g = pred[(pred.event == ev0) & (pred.h == 48)]
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for m in ["Persistence", "Raw CAMS"] + VARIANTS:
    p, r, _ = precision_recall_curve(g["y"], g[f"score|{m}"])
    ax.plot(r, p, lw=1.2, ls="--" if m in ("Persistence", "Raw CAMS") else "-", label=m)
ax.axhline(g["y"].mean(), color="grey", lw=0.8, ls=":", label=f"Random ({g['y'].mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall, event at h = 48")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "pr_curves_h48.png", dpi=200); plt.close(fig)

s = by_season[(by_season.event == ev0) & by_season.model.isin(["Persistence", "Raw CAMS"] + VARIANTS)]
piv = s.pivot_table(index="model", columns=["h", "season"], values="recall")
fig, ax = plt.subplots(figsize=(8, 3.8))
piv.T.plot.bar(ax=ax, width=0.8)
ax.set_ylabel("Recall"); ax.set_xlabel("(horizon, season)"); ax.set_title("Event recall by season (cost 10:1)")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "recall_by_season.png", dpi=200); plt.close(fig)

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")
