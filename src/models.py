"""Model factories. Hyperparameters tuned by Optuna can be passed in for LightGBM."""
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42

LGBM_DEFAULT = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)


def lgbm_regressor(params=None):
    """`params` may hold log_target=True (fit on log1p(PM2.5), predict back with expm1)."""
    params = dict(params or {})
    log_target = params.pop("log_target", False)
    model = LGBMRegressor(**{**LGBM_DEFAULT, **params}, random_state=SEED, verbose=-1)
    if log_target:
        return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model


def lgbm_classifier(params=None, scale_pos_weight=1.0):
    return LGBMClassifier(**{**LGBM_DEFAULT, **(params or {})}, scale_pos_weight=scale_pos_weight,
                          random_state=SEED, verbose=-1)


def xgb_regressor():
    return XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1)


def ridge_regressor():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def rf_regressor():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=5,
                                               max_features=0.5, n_jobs=-1, random_state=SEED))
