# %% [markdown]
# # Step 5f: RQ2 check on the same test hours
# In Step 3 the random 80/20 split was scored on all hours since Aug 2022, while the rolling-origin
# backtest is scored on Aug 2023 - Mar 2025. This check scores both on exactly the same hours, for the
# tuned LightGBM + CAMS and for the best model (Ridge + CAMS), so the gap cannot come from a different
# test period.
# Run after 03:   %run notebooks/11_rq2_same_period.py   (about 2 min)
# Output: results/rq2_same_period.csv

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path.cwd()))
from src.config import CAMS_START, HORIZONS, RESULTS, ROOT
from src.evaluation import regression_metrics
from src.features import add_targets, feature_list
from src.models import SEED, lgbm_regressor, ridge_regressor

BEST = json.loads((RESULTS / "optuna_best_params.json").read_text())["best_params"]
MODELS = {"LightGBM + CAMS (tuned)": lambda: lgbm_regressor(BEST), "Ridge + CAMS": ridge_regressor}
df = add_targets(pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["time"], index_col="time"))
reg = pd.read_csv(ROOT / "data" / "predictions_regression.csv", parse_dates=["time"], index_col="time")

rows = []
for h in HORIZONS:
    y, cols = df[f"y_pm25_h{h}"], feature_list(h, True)
    ok = (df.index >= CAMS_START) & y.notna() & df["pm25"].notna() & df[f"cams_h{h}"].notna()
    X, yy = df.loc[ok, cols], y[ok]
    test_hours = reg[reg.h == h].index.intersection(X.index)
    roll = reg[reg.h == h].loc[test_hours]
    for name, make in MODELS.items():
        p = pd.Series(np.nan, index=X.index)
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
            p.iloc[te] = np.maximum(make().fit(X.iloc[tr], yy.iloc[tr]).predict(X.iloc[te]), 0)
        rnd = regression_metrics(yy[test_hours], p[test_hours])
        rol = regression_metrics(roll["y"], roll[name])
        rows.append({"h": h, "model": name, "hours": len(test_hours),
                     "random_MAE": rnd["MAE"], "random_R2": rnd["R2"],
                     "rolling_MAE": rol["MAE"], "rolling_R2": rol["R2"],
                     "MAE_underestimated_by_pct": 100 * (rol["MAE"] - rnd["MAE"]) / rol["MAE"]})
    print(f"h={h} done")

out = pd.DataFrame(rows)
out.round(3).to_csv(RESULTS / "rq2_same_period.csv", index=False)
pd.set_option("display.width", 200)
print("\nRQ2 on the same test hours (Aug 2023 - Mar 2025):")
print(out.round(2).to_string(index=False))
