# %% [markdown]
# # Step 4: LSTM comparison and SHAP interpretation
# Run after 03_regression.py (it reads data/predictions_regression.csv):
#     %run notebooks/05_lstm_shap.py
# Use a GPU runtime (Runtime -> Change runtime type -> T4 GPU): about 10 min. On CPU it takes longer.
#
# Part A: an LSTM (Keras) trained on the same 10 rolling-origin folds and scored on exactly the same
#         test hours as the other models, as the proposal's recurrent comparison.
# Part B: SHAP values for LightGBM + CAMS at 24/48/72 h (interpretation, not evaluation).
# Outputs: results/lstm_comparison.csv, lstm_dm_tests.csv, shap_importance.csv,
#          figures/lstm_comparison.png, shap_summary_h48.png, shap_importance_by_horizon.png,
#          shap_dependence_cams_h48.png

# %%
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path.cwd()))
from src.config import CAMS_START, HORIZONS, RESULTS, ROOT
from src.evaluation import diebold_mariano, make_folds, regression_metrics, split_masks
from src.features import add_targets, feature_list
from src.models import SEED, lgbm_regressor

QUICK = os.getenv("PM25_QUICK") == "1"
FIRST_TEST = pd.Timestamp("2023-08-01")
SEQ = 48                                     # hours of history the LSTM sees
EPOCHS = 3 if QUICK else 60
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
t_start = time.time()
tf.keras.utils.set_random_seed(SEED)
print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU") or "none")

df = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"], index_col="time")
df = add_targets(df)

# %% [markdown]
# ## A1. Inputs: a 48-hour sequence per hour, plus the CAMS forecasts for the three target hours
# PM2.5 and CAMS are log-transformed; everything is standardised with statistics from data before
# the first test fold only. Missing values become 0 after scaling, with a "PM2.5 missing" flag.

# %%
SEQ_COLS = ["pm25", "cams_pm25", "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "surface_pressure", "wind_speed_10m", "wind_sin", "wind_cos", "boundary_layer_height",
            "log_ventilation", "precipitation", "hour_sin", "hour_cos", "month_sin", "month_cos"]
STATIC_COLS = [f"cams_h{h}" for h in HORIZONS] + [f"cams_mean24_h{h}" for h in HORIZONS] + ["smog_season"]
LOG_COLS = ["pm25", "cams_pm25"] + [f"cams_h{h}" for h in HORIZONS] + [f"cams_mean24_h{h}" for h in HORIZONS]
Y_COLS = [f"y_pm25_h{h}" for h in HORIZONS]

d = df[df.index >= pd.Timestamp(CAMS_START) - pd.Timedelta(hours=SEQ)].copy()
d[LOG_COLS] = np.log1p(d[LOG_COLS])
pre = d.index < FIRST_TEST


def scale(cols):
    x = d[cols]
    return ((x - x[pre].mean()) / x[pre].std().replace(0, 1)).fillna(0).astype("float32")


seq_frame = scale(SEQ_COLS)
seq_frame["pm25_missing"] = df.loc[d.index, "pm25"].isna().astype("float32")
A = seq_frame.to_numpy()
static = scale(STATIC_COLS).to_numpy()
Y = (d[Y_COLS] / 100.0).to_numpy("float32")            # targets in units of 100 µg/m³

windows = sliding_window_view(A, SEQ, axis=0)          # (n - SEQ + 1, features, SEQ)
pos = np.arange(SEQ - 1, len(d))                        # rows that have a full history window
pos = pos[d.index[pos] >= pd.Timestamp(CAMS_START)]
print(f"Sequences: {len(pos)} x {SEQ} h x {A.shape[1]} features, static {static.shape[1]}")


def batch(rows):
    return [windows[rows - SEQ + 1].transpose(0, 2, 1), static[rows]], Y[rows]


def masked_mae(y_true, y_pred):
    ok = tf.math.logical_not(tf.math.is_nan(y_true))
    return tf.reduce_mean(tf.abs(tf.boolean_mask(y_true, ok) - tf.boolean_mask(y_pred, ok)))


def build_lstm():
    s_in = tf.keras.Input((SEQ, A.shape[1]))
    x = tf.keras.layers.LSTM(64)(s_in)
    x = tf.keras.layers.Dropout(0.2)(x)
    c_in = tf.keras.Input((static.shape[1],))
    z = tf.keras.layers.Concatenate()([x, c_in])
    z = tf.keras.layers.Dense(64, activation="relu")(z)
    out = tf.keras.layers.Dense(len(HORIZONS))(z)        # direct multi-horizon output: 24, 48, 72 h
    model = tf.keras.Model([s_in, c_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=masked_mae)
    return model


# %% [markdown]
# ## A2. Rolling-origin training (same folds as Step 3). Training targets end before the test block
# (gap = 72 h, the longest horizon). The last 10% of each training set is used for early stopping.

# %%
idx = d.index
folds = make_folds(df.index[df.index >= CAMS_START], FIRST_TEST)
if QUICK:
    folds = folds[:3]
lstm_preds = []
for k, (a, b) in enumerate(folds):
    tr_mask, te_mask = split_masks(idx[pos], a, b, max(HORIZONS), train_start=CAMS_START)
    tr = pos[tr_mask & ~np.isnan(Y[pos]).all(axis=1)]
    te = pos[te_mask]
    if len(te) == 0:
        continue
    cut = int(len(tr) * 0.9)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + k)
    model = build_lstm()
    hist = model.fit(*batch(tr[:cut]), validation_data=tuple(batch(tr[cut:])), epochs=EPOCHS,
                     batch_size=256, verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    p = np.maximum(model.predict(batch(te)[0], batch_size=1024, verbose=0) * 100, 0)
    lstm_preds.append(pd.DataFrame(p, index=idx[te], columns=list(HORIZONS)))
    print(f"fold {k} ({a:%Y-%m}): train {len(tr)}, test {len(te)}, epochs {len(hist.history['loss'])}, "
          f"val MAE {100 * min(hist.history['val_loss']):.1f} [{(time.time() - t_start) / 60:.1f} min]")
lstm = pd.concat(lstm_preds)

# %% [markdown]
# ## A3. Compare on exactly the same test hours as Step 3

# %%
reg = pd.read_csv(ROOT / "data" / "predictions_regression.csv", parse_dates=["time"], index_col="time")
reg = reg[reg.fold.isin(range(len(folds)))]
reg["LSTM"] = [lstm.at[t, h] if t in lstm.index else np.nan for t, h in zip(reg.index, reg.h)]
reg = reg.dropna(subset=["LSTM"])
COMPARE = ["Persistence", "Raw CAMS", "Ridge + CAMS", "Random Forest + CAMS", "XGBoost + CAMS",
           "LightGBM + CAMS", "LightGBM + CAMS (tuned)", "LSTM"]
smog = lambda i: i.month.isin([10, 11, 12, 1, 2])

rows, dm_rows = [], []
for h, g in reg.groupby("h"):
    g = g.sort_index()
    season = np.where(smog(g.index + pd.Timedelta(hours=int(h))), "Oct-Feb", "Mar-Sep")
    for m in COMPARE:
        r = regression_metrics(g["y"], g[m])
        for s in ["Oct-Feb", "Mar-Sep"]:
            r[f"MAE_{s}"] = regression_metrics(g["y"][season == s], g[m][season == s])["MAE"]
        rows.append({"h": h, "model": m, **r})
        if m != "LSTM":
            stat, pv = diebold_mariano(g["LSTM"] - g["y"], g[m] - g["y"], int(h))
            dm_rows.append({"h": h, "LSTM_vs": m, "MAE_LSTM": np.mean(np.abs(g["LSTM"] - g["y"])),
                            "MAE_other": np.mean(np.abs(g[m] - g["y"])), "DM_stat": stat, "p_value": pv})
comp, dm = pd.DataFrame(rows), pd.DataFrame(dm_rows)
comp.round(3).to_csv(RESULTS / "lstm_comparison.csv", index=False)
dm.round(4).to_csv(RESULTS / "lstm_dm_tests.csv", index=False)
pd.set_option("display.width", 200)
for h in HORIZONS:
    print(f"\n=== h = {h}: same test hours as Step 3 ===")
    print(comp[comp.h == h].drop(columns="h").sort_values("MAE").round(2).to_string(index=False))
print("\nDiebold-Mariano: LSTM vs others (negative = LSTM better)")
print(dm.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.8))
piv = comp.pivot(index="model", columns="h", values="MAE").loc[COMPARE]
piv.plot.bar(ax=ax, width=0.8)
ax.set_ylabel("MAE (µg/m³)"); ax.set_xlabel(""); ax.set_title("LSTM vs other models (same test hours)")
ax.legend(title="Horizon (h)", fontsize=8); plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "lstm_comparison.png", dpi=200); plt.close(fig)

# %% [markdown]
# ## B. SHAP for LightGBM + CAMS
# Trained on all short-window data (interpretation only; the accuracy numbers come from the backtest).

# %%
imp = {}
N_SHAP = 500 if QUICK else 3000
for h in HORIZONS:
    y = df[f"y_pm25_h{h}"]
    rows_ok = (df.index >= CAMS_START) & y.notna()
    cols = feature_list(h, True)
    model = lgbm_regressor().fit(df.loc[rows_ok, cols], y[rows_ok])
    X = df.loc[rows_ok, cols].sample(min(N_SHAP, rows_ok.sum()), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(X)
    imp[h] = pd.Series(np.abs(sv).mean(axis=0), index=[c.replace(f"_h{h}", "") for c in cols])
    if h == 48:
        shap.summary_plot(sv, X, max_display=15, show=False)
        plt.title("SHAP values, LightGBM + CAMS, h = 48"); plt.tight_layout()
        plt.savefig(FIG / "shap_summary_h48.png", dpi=200, bbox_inches="tight"); plt.close("all")
        shap.dependence_plot("cams_h48", sv, X, interaction_index=None, show=False)
        plt.title("Effect of the CAMS forecast on the prediction, h = 48"); plt.tight_layout()
        plt.savefig(FIG / "shap_dependence_cams_h48.png", dpi=200, bbox_inches="tight"); plt.close("all")

imp = pd.DataFrame(imp)
imp.columns = [f"mean_abs_shap_h{h}" for h in HORIZONS]
imp = imp.sort_values(f"mean_abs_shap_h{HORIZONS[0]}", ascending=False)
imp.round(3).to_csv(RESULTS / "shap_importance.csv", index_label="feature")
print("\nMean |SHAP| (µg/m³), top 15:")
print(imp.head(15).round(2).to_string())

fig, ax = plt.subplots(figsize=(6.5, 5))
imp.head(12)[::-1].plot.barh(ax=ax, width=0.8)
ax.set_xlabel("Mean |SHAP| (µg/m³)"); ax.set_title("Feature importance by horizon")
ax.legend([f"{h} h" for h in HORIZONS], fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "shap_importance_by_horizon.png", dpi=200); plt.close(fig)

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")
