"""Performance metrics, intervals and paired comparisons.

Survival convention used everywhere in exprkit: a risk score is a log hazard.
Higher risk means higher hazard, i.e. shorter expected survival.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, mean_absolute_error, r2_score, roc_auc_score)


# Survival -----------------------------------------------------------------------

def harrell_c(time, event, risk) -> float:
    """Harrell's C (Harrell, Lee and Mark 1996, Stat Med 15:361).

    Comparable pair: time_i < time_j and event_i = 1 (the later patient may be
    censored). Concordant if risk_i > risk_j; ties in risk count one half;
    pairs with tied times are skipped. Matches
    ``sksurv.metrics.concordance_index_censored`` (tested).
    """
    t, e, r = (np.asarray(a, dtype=float) for a in (time, event, risk))
    num = den = 0.0
    ev = np.flatnonzero(e == 1)
    for start in range(0, len(ev), 512):
        i = ev[start:start + 512]
        comp = t[i][:, None] < t[None, :]
        diff = r[i][:, None] - r[None, :]
        num += (comp & (diff > 0)).sum() + 0.5 * (comp & (diff == 0)).sum()
        den += comp.sum()
    return float(num / den) if den else float("nan")


def cox_neg_log_partial_likelihood(time, event, risk) -> float:
    """Slow reference Breslow negative log partial likelihood per event (for tests)."""
    t, e, r = (np.asarray(a, dtype=float) for a in (time, event, risk))
    total = 0.0
    for i in np.flatnonzero(e == 1):
        total += r[i] - np.log(np.exp(r[t >= t[i]]).sum())
    return float(-total / max(e.sum(), 1))


# Classification -------------------------------------------------------------------

def classification_metrics(y_true, proba: np.ndarray, classes: list, positive=None) -> dict:
    """Metrics from predicted probabilities (columns ordered as ``classes``).

    Binary: ROC AUC and average precision for the positive class, average
    precision for the other class, balanced accuracy and per class recall at
    the default decision (highest probability, i.e. 0.5), confusion counts.
    Multiclass: one versus rest macro ROC AUC, balanced accuracy, per class recall.
    """
    y = np.asarray(y_true).astype(str)
    classes = [str(c) for c in classes]
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    out: dict = {"n": int(len(y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                 "accuracy": float(np.mean(pred == y))}
    for c in classes:
        m = y == c
        out[f"recall_{c}"] = float(np.mean(pred[m] == c)) if m.any() else float("nan")
    if len(classes) == 2:
        pos = str(positive) if positive is not None else classes[1]
        k = classes.index(pos)
        yb = (y == pos).astype(int)
        out["roc_auc"] = float(roc_auc_score(yb, proba[:, k])) if 0 < yb.sum() < len(yb) else float("nan")
        out[f"average_precision_{pos}"] = float(average_precision_score(yb, proba[:, k]))
        neg = classes[1 - k]
        out[f"average_precision_{neg}"] = float(average_precision_score(1 - yb, proba[:, 1 - k]))
        cm = confusion_matrix(y, pred, labels=classes)
        out["confusion_matrix"] = {"labels": classes, "rows_true_cols_pred": cm.tolist()}
    else:
        present = [c for c in classes if (y == c).any()]
        if len(present) == len(classes):
            out["roc_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes))
        else:
            out["roc_auc"] = float("nan")
        out["confusion_matrix"] = {"labels": classes,
                                   "rows_true_cols_pred": confusion_matrix(y, pred, labels=classes).tolist()}
    return out


def auc_score(y_true, proba, classes, positive=None) -> float:
    y = np.asarray(y_true).astype(str)
    classes = [str(c) for c in classes]
    if len(classes) == 2:
        pos = str(positive) if positive is not None else classes[1]
        yb = (y == pos).astype(int)
        if yb.sum() in (0, len(yb)):
            return float("nan")
        return float(roc_auc_score(yb, proba[:, classes.index(pos)]))
    try:
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes))
    except ValueError:
        return float("nan")


def exact_proportion_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper and Pearson exact interval for a proportion k/n.

    Useful for the recall of a small class: 10 of 10 normals correct still
    only bounds the true recall above about 0.69 at 95%.
    """
    if n == 0:
        return float("nan"), float("nan")
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=level, method="exact")
    return float(ci.low), float(ci.high)


# Regression -------------------------------------------------------------------------

def regression_metrics(y_true, pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(pred, float)
    return {"n": int(len(y)), "r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)),
            "spearman": float(stats.spearmanr(y, p).statistic), "pearson": float(stats.pearsonr(y, p).statistic)}


# Intervals and comparisons ------------------------------------------------------------

def bootstrap_interval(metric, n: int, n_boot: int = 1000, seed: int = 0, strata=None,
                       level: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap. ``metric(idx)`` returns the statistic on rows ``idx``.

    With ``strata`` (e.g. class labels) rows are resampled within each stratum,
    so every replicate keeps the original class sizes; this matters when a
    class is small.
    """
    rng = np.random.default_rng(seed)
    full = metric(np.arange(n))
    if strata is not None:
        groups = [np.flatnonzero(np.asarray(strata) == s) for s in np.unique(strata)]
    vals = []
    for _ in range(n_boot):
        if strata is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        vals.append(metric(idx))
    a = (1 - level) / 2
    lo, hi = np.nanpercentile(vals, [100 * a, 100 * (1 - a)])
    return float(full), float(lo), float(hi)


def paired_bootstrap(metric_a, metric_b, n: int, n_boot: int = 1000, seed: int = 0, strata=None) -> dict:
    """Difference metric_b minus metric_a on the same resampled rows."""
    rng = np.random.default_rng(seed)
    if strata is not None:
        groups = [np.flatnonzero(np.asarray(strata) == s) for s in np.unique(strata)]
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n) if strata is None else np.concatenate([rng.choice(g, len(g)) for g in groups])
        d.append(metric_b(idx) - metric_a(idx))
    full = metric_b(np.arange(n)) - metric_a(np.arange(n))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"difference": float(full), "ci_low": float(lo), "ci_high": float(hi),
            "fraction_of_resamples_positive": float(np.nanmean(np.array(d) > 0)), "method": "paired bootstrap"}


def corrected_resampled_t(values, n_train: float, n_test: float, null: float = 0.0,
                          bounds: tuple[float, float] | None = None, alternative: str = "two-sided") -> dict:
    """Mean of per fold values (or per fold differences) with the variance
    correction of Nadeau and Bengio (2003, Machine Learning 52:239) for
    overlapping training sets in repeated cross validation:

        var_corrected = (1/J + n_test/n_train) * sample variance, J = number of folds in total.

    Returns the mean, a 95% interval and a p value against ``null``:
    ``alternative="greater"`` tests whether the mean is above ``null`` (used
    for "better than chance"), ``"two-sided"`` whether it differs (used for
    differences between two models).
    ``bounds`` clips the interval to the metric's range (e.g. 0 to 1 for AUC).
    """
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    J = len(v)
    m = float(v.mean())
    if J < 2:
        return {"mean": m, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "n_folds": J}
    se = np.sqrt((1 / J + n_test / n_train) * v.var(ddof=1))
    tcrit = stats.t.ppf(0.975, J - 1)
    if se == 0:
        p = 0.0 if m != null else 1.0
    else:
        tstat = (m - null) / se
        p = float(stats.t.sf(tstat, J - 1)) if alternative == "greater" else float(2 * stats.t.sf(abs(tstat), J - 1))
    lo, hi = m - tcrit * se, m + tcrit * se
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return {"mean": m, "ci_low": float(lo), "ci_high": float(hi), "p_value": p, "alternative": alternative,
            "n_folds": J, "method": "corrected resampled t (Nadeau and Bengio 2003)"}
