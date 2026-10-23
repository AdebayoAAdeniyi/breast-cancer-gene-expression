"""Resampling, out of fold predictions, permutation tests and design checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score

from .metrics import auc_score, harrell_c
from .models import final_pipeline, fit_estimator


# Splits -------------------------------------------------------------------------

def stratify_labels(task: str, y) -> np.ndarray:
    if task == "classification":
        return np.asarray(y).astype(str)
    if task == "survival":
        return np.asarray(y["event"]).astype(int)
    q = pd.qcut(pd.Series(np.asarray(y, float)).rank(method="first"), 5, labels=False)
    return q.to_numpy()


def make_splits(task: str, y, groups, scheme: dict, seed: int,
                split_groups=None) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """List of (repeat, fold, train_idx, test_idx).

    ``scheme`` is ``{type: repeated_cv, n_splits: 5, n_repeats: 10}`` or
    ``{type: holdout, test_fraction: 0.2}``. Splits are stratified (class,
    event or quintile of the target) and grouped when ``groups`` is given.
    """
    s = stratify_labels(task, y)
    z = np.zeros(len(s))
    kind = scheme.get("type", "repeated_cv")
    out = []
    if kind == "repeated_cv":
        k, r = int(scheme.get("n_splits", 5)), int(scheme.get("n_repeats", 10))
        if task == "classification":
            smallest = pd.Series(s).value_counts().min()
            if smallest < k:
                raise ValueError(f"the smallest class has {smallest} samples, fewer than n_splits={k}")
        for rep in range(r):
            cv = (StratifiedGroupKFold if groups is not None else StratifiedKFold)(
                n_splits=k, shuffle=True, random_state=seed + rep)
            for f, (tr, te) in enumerate(cv.split(z, s, groups)):
                out.append((rep, f, tr, te))
    elif kind == "leave_one_group_out":
        # e.g. leave one study out: train on all other studies, test on the held out one
        if split_groups is None:
            raise ValueError("leave_one_group_out needs evaluation.column (e.g. dataset)")
        sg = np.asarray(split_groups).astype(str)
        for f, gname in enumerate(sorted(set(sg))):
            te = np.flatnonzero(sg == gname)
            tr = np.flatnonzero(sg != gname)
            if task == "classification" and len(set(np.asarray(y)[tr].astype(str))) < 2:
                raise ValueError(f"leaving out '{gname}' leaves a single class for training")
            out.append((0, f, tr, te))
    elif kind == "holdout":
        k = int(round(1 / float(scheme.get("test_fraction", 0.2))))
        cv = (StratifiedGroupKFold if groups is not None else StratifiedKFold)(n_splits=k, shuffle=True,
                                                                              random_state=seed)
        tr, te = next(cv.split(z, s, groups))
        out.append((0, 0, tr, te))
    else:
        raise ValueError("evaluation type must be 'repeated_cv', 'holdout' or 'leave_one_group_out'")
    if groups is not None:
        g = np.asarray(groups)
        for _, _, tr, te in out:
            shared = set(g[tr]) & set(g[te])
            if shared:
                raise AssertionError(f"groups {sorted(shared)[:5]} appear in both training and test rows")
    return out


# Predictions -----------------------------------------------------------------------

def predict_output(task: str, est, X) -> np.ndarray:
    if task == "classification":
        return est.predict_proba(X)
    return np.asarray(est.predict(X), dtype=float)


def fold_score(task: str, y_te, pred, classes=None, positive=None) -> float:
    if task == "classification":
        return auc_score(y_te, pred, classes, positive)
    if task == "survival":
        return harrell_c(y_te["time"], y_te["event"], pred)
    return float(r2_score(y_te, pred))


def risk_cutpoints(risk: np.ndarray, n_groups: int = 3) -> list[float]:
    return np.quantile(risk, np.linspace(0, 1, n_groups + 1)[1:-1]).tolist()


def assign_risk_groups(risk: np.ndarray, cutpoints: list[float]) -> np.ndarray:
    names = ["low", "middle", "high"] if len(cutpoints) == 2 else [f"q{i + 1}" for i in range(len(cutpoints) + 1)]
    return np.asarray(names)[np.searchsorted(cutpoints, risk, side="right")]


def cross_validate(task: str, build, X: pd.DataFrame, y, groups, splits, classes=None, positive=None,
                   top_k_features: int = 50) -> dict:
    """Fit a fresh estimator on each training split and predict the test rows.

    Returns per fold scores, out of fold predictions per repeat, the top
    features selected in each fold (for stability), and for survival the
    risk group of each test row using cut points from its own training rows.
    """
    from .models import coefficients
    n = len(X)
    rows, preds, groups_out, top = [], {}, {}, []
    for rep, fold, tr, te in splits:
        est = build()
        fit_estimator(est, X.iloc[tr], y[tr], None if groups is None else np.asarray(groups)[tr])
        p = predict_output(task, est, X.iloc[te])
        if rep not in preds:
            shape = (n, len(classes)) if task == "classification" else (n,)
            preds[rep] = np.full(shape, np.nan)
            groups_out[rep] = np.full(n, None, dtype=object)
        preds[rep][te] = p
        if task == "survival":
            cut = risk_cutpoints(predict_output(task, est, X.iloc[tr]))
            groups_out[rep][te] = assign_risk_groups(p, cut)
        row = {"repeat": rep, "fold": fold, "n_train": len(tr), "n_test": len(te),
               "score": fold_score(task, y[te], p, classes, positive)}
        if task == "classification":
            yt = np.asarray(y[te]).astype(str)
            pred_cls = np.asarray(classes).astype(str)[np.argmax(p, axis=1)]
            for c in classes:
                mc = yt == str(c)
                row[f"n_{c}"] = int(mc.sum())
                row[f"recall_{c}"] = float(np.mean(pred_cls[mc] == str(c))) if mc.any() else float("nan")
        try:
            from .models import best_params
            row.update({f"param_{k}": str(v) for k, v in best_params(est).items()})
        except Exception:
            pass
        rows.append(row)
        coef = coefficients(est)
        if coef is not None:
            nz = coef[coef.abs() > 0]
            top.append(set(nz.abs().sort_values(ascending=False).index[:top_k_features]))
    stability = None
    if top:
        counts = pd.Series([f for s in top for f in s]).value_counts()
        stability = (counts / len(top)).rename("fraction_of_fits_in_top").to_frame()
    return {"folds": pd.DataFrame(rows), "preds": preds, "risk_groups": groups_out, "stability": stability}


# Permutation test ----------------------------------------------------------------------

def permutation_test(task: str, build_fixed, X, y, groups, splits, classes=None, positive=None,
                     n_permutations: int = 100, seed: int = 0) -> dict:
    """Refit the pipeline on the same splits with the target rows shuffled.

    Score per run: mean fold score. The pipeline uses fixed hyperparameters
    (those chosen on the full data) to keep the cost manageable; the observed
    score is recomputed with the same fixed pipeline so both sides match.
    p = (1 + number of permuted scores >= observed) / (1 + n_permutations)
    (Ojala and Garriga 2010, JMLR 11:1833).
    """
    def run(yy):
        scores = []
        for _, _, tr, te in splits:
            est = build_fixed()
            fit_estimator(est, X.iloc[tr], yy[tr])
            scores.append(fold_score(task, yy[te], predict_output(task, est, X.iloc[te]), classes, positive))
        return float(np.nanmean(scores))

    rng = np.random.default_rng(seed)
    observed = run(y)
    perm = []
    for _ in range(n_permutations):
        perm.append(run(y[rng.permutation(len(y))]))
    perm = np.asarray(perm)
    return {"observed_score_fixed_params": observed, "permuted_mean": float(np.nanmean(perm)),
            "permuted_95th_percentile": float(np.nanpercentile(perm, 95)),
            "p_value": float((1 + np.sum(perm >= observed)) / (1 + len(perm))),
            "n_permutations": int(n_permutations),
            "note": "rows of the target shuffled jointly (time and event together for survival); groups not "
                    "preserved under permutation"}


# Design checks --------------------------------------------------------------------------

def confounding_table(target: pd.Series, batch: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Cross table of target classes by batch, and warnings when a class occurs
    in only one batch while other batches exist (class and batch confounded)."""
    tab = pd.crosstab(target.astype(str), batch.astype(str))
    warnings = []
    if tab.shape[1] > 1:
        for cls, row in tab.iterrows():
            present = row[row > 0].index.tolist()
            if len(present) == 1:
                warnings.append(f"class '{cls}' occurs only in batch '{present[0]}': a model can separate it by "
                                f"recognising the batch. Evaluate within one batch or check batch separability.")
    return tab, warnings


def batch_separability(X: pd.DataFrame, batch: pd.Series, n_splits: int = 5, seed: int = 0,
                       top_k_variance: int = 2000) -> dict:
    """Cross validated ROC AUC for predicting which of two batches a sample came from.

    Use samples of a single biological class (e.g. cancer only), so any
    separability is due to the batch (study, platform, lab). An AUC near 1 means
    pooling across these batches will confound any class that occurs in only one.
    """
    from .models import build_pipeline
    b = batch.astype(str)
    top2 = b.value_counts().index[:2].tolist()
    if len(top2) < 2:
        raise ValueError("need at least two batches")
    m = b.isin(top2).to_numpy()
    Xs, bs = X.loc[m], b[m]
    y = (bs == top2[1]).astype(int).to_numpy()
    pipe = build_pipeline("classification", {"type": "logistic"}, list(Xs.columns), [], [],
                          {"top_k_variance": top_k_variance}, seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(pipe, Xs, y, cv=cv, method="predict_proba")[:, 1]
    return {"batches": top2, "n_per_batch": bs.value_counts().to_dict(), "auc_predicting_batch": float(roc_auc_score(y, p)),
            "genes_used": int(Xs.shape[1]),
            "interpretation": "AUC near 0.5: batches not distinguishable. AUC near 1: strong batch effect."}
