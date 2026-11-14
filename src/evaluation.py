"""Rolling-origin folds, metrics, cost-based thresholds and the Diebold-Mariano test."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


# ----------------------------------------------------------------------------- folds
def make_folds(index, first_test, months=2):
    """Expanding-window rolling-origin folds: each test block is `months` long."""
    end = index.max() + pd.Timedelta(hours=1)
    folds, start = [], pd.Timestamp(first_test)
    while start < end:
        stop = min(start + pd.DateOffset(months=months), end)
        folds.append((start, stop))
        start = stop
    return folds


def split_masks(index, test_start, test_end, horizon, train_start=None):
    """Train rows must have their target strictly before the test block (a gap of `horizon` hours)."""
    train = index < (test_start - pd.Timedelta(hours=horizon))
    if train_start is not None:
        train &= index >= pd.Timestamp(train_start)
    test = (index >= test_start) & (index < test_end)
    return train, test


# ----------------------------------------------------------------------------- regression
def regression_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(y) < 10:
        return dict(n=len(y), MAE=np.nan, RMSE=np.nan, R2=np.nan, MAPE=np.nan)
    err = p - y
    big = y >= 10                      # MAPE is unstable near zero; computed on y >= 10 ug/m3
    return dict(
        n=len(y),
        MAE=np.mean(np.abs(err)),
        RMSE=np.sqrt(np.mean(err ** 2)),
        R2=1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
        MAPE=100 * np.mean(np.abs(err[big]) / y[big]),
    )


def diebold_mariano(e1, e2, h, power=1):
    """DM test with the Harvey-Leybourne-Newbold correction.
    Negative statistic = model 1 has lower loss. Returns (statistic, two-sided p-value)."""
    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 50:
        return np.nan, np.nan
    dc = d - d.mean()
    lags = max(1, h - 1)
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lags + 1)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * stats.t.sf(abs(dm), df=n - 1)


def block_bootstrap_diff(loss_a, loss_b, times, n_boot=2000, seed=42):
    """Day-block bootstrap of mean(loss_a - loss_b). Whole days are resampled so that
    autocorrelation within a day is kept. Returns (mean difference, 2.5%, 97.5%)."""
    d = pd.Series(np.asarray(loss_a, float) - np.asarray(loss_b, float),
                  index=pd.DatetimeIndex(times)).dropna()
    g = d.groupby(d.index.normalize())
    s, c = g.sum().to_numpy(), g.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi


# ----------------------------------------------------------------------------- classification
def expected_cost(y, pred, fn_cost=10, fp_cost=1):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return fn_cost * fn + fp_cost * fp


def best_threshold(y, prob, fn_cost=10):
    grid = np.linspace(0.02, 0.98, 49)
    costs = [expected_cost(y, (prob >= t).astype(int), fn_cost) for t in grid]
    return float(grid[int(np.argmin(costs))])


def classification_metrics(y, prob, threshold=None, fn_cost=10, pred=None):
    """`prob` is any score (probability or a PM2.5 value) used for PR-AUC / ROC-AUC.
    Labels come from `pred` if given (e.g. per-fold thresholds), else prob >= threshold."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int) if pred is None else np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(
        n=len(y), positives=int(y.sum()), threshold=np.nan if threshold is None else threshold,
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0) if y.sum() > 0 else np.nan,
        F1=f1_score(y, pred, zero_division=0),
        PR_AUC=average_precision_score(y, prob) if both else np.nan,
        PR_AUC_random=y.mean(),
        ROC_AUC=roc_auc_score(y, prob) if both else np.nan,
        accuracy=accuracy_score(y, pred),
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
        cost=fn_cost * fn + fp,
        false_alarms_per_30d=fp / (len(y) / 720),
    )
