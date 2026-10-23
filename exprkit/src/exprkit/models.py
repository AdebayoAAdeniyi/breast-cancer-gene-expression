"""Model pipelines for the three supported tasks.

Every model is a scikit-learn Pipeline:

    ColumnTransformer
      genes:       ExpressionFilter -> median imputation -> standardisation
      numeric covariates:     median imputation -> standardisation
      categorical covariates: most frequent imputation -> one hot encoding
    model

All preprocessing is fitted on the training rows of each fold. When the
config gives a grid of hyperparameters, the pipeline is wrapped in
GridSearchCV with an inner cross validation on the training rows only, so
the outer evaluation is a nested cross validation.

Models
------
classification  logistic (L2, class weighted), elasticnet_logistic, random_forest, mlp
regression      ridge, elasticnet, random_forest
survival        cox (ridge penalised Cox), coxnet (elastic net Cox), deepsurv (needs torch)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .filtering import ExpressionFilter

MODELS = {
    "classification": ("logistic", "elasticnet_logistic", "random_forest", "mlp"),
    "regression": ("ridge", "elasticnet", "random_forest"),
    "survival": ("cox", "coxnet", "deepsurv"),
}


def make_model(task: str, kind: str, seed: int = 0):
    if kind not in MODELS.get(task, ()):
        raise ValueError(f"model '{kind}' is not available for task '{task}'. Choose from {MODELS.get(task)}")
    if task == "classification":
        if kind == "logistic":
            return LogisticRegression(C=0.1, class_weight="balanced", max_iter=10000)
        if kind == "elasticnet_logistic":
            return LogisticRegression(C=0.1, l1_ratio=0.5, solver="saga", class_weight="balanced", max_iter=20000)
        if kind == "random_forest":
            return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000, random_state=seed)
    if task == "regression":
        if kind == "ridge":
            return Ridge(alpha=10.0)
        if kind == "elasticnet":
            return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000)
        return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
    if kind == "cox":
        return CoxPHSurvivalAnalysis(alpha=1e-2)
    if kind == "coxnet":
        return CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[0.1], max_iter=100000)
    from .deepsurv import DeepSurvEstimator
    return DeepSurvEstimator(seed=seed)


def _translate(task: str, kind: str, params: dict) -> dict:
    """Config names to estimator names. For coxnet, ``alpha`` becomes ``alphas=[alpha]``."""
    out = {}
    for k, v in (params or {}).items():
        if task == "survival" and kind == "coxnet" and k == "alpha":
            out["alphas"] = [[a] for a in v] if isinstance(v, list) else [v]
        else:
            out[k] = v
    return out


def inner_cv(task: str, n_splits: int, seed: int, grouped: bool):
    if task == "classification":
        return (StratifiedGroupKFold if grouped else StratifiedKFold)(n_splits=n_splits, shuffle=True, random_state=seed)
    if task == "survival":
        return EventStratifiedCV(n_splits, seed, grouped)
    if grouped:
        return GroupKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


class EventStratifiedCV:
    """Folds stratified on the event indicator of a survival target (optionally grouped)."""

    def __init__(self, n_splits=3, seed=0, grouped=False):
        self.n_splits, self.seed, self.grouped = n_splits, seed, grouped

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        ev = np.asarray(y["event"]).astype(int)
        cv = (StratifiedGroupKFold if self.grouped and groups is not None else StratifiedKFold)(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        return cv.split(np.zeros(len(ev)), ev, groups if self.grouped else None)


def scoring_for(task: str, n_classes: int = 2):
    if task == "classification":
        return "roc_auc" if n_classes == 2 else "roc_auc_ovr"
    if task == "regression":
        return "r2"
    return None  # survival estimators score with Harrell's C


def build_pipeline(task: str, model_cfg: dict, gene_cols: list[str], numeric_covariates: list[str],
                   categorical_covariates: list[str], filter_cfg: dict | None, seed: int = 0,
                   n_classes: int = 2, grouped: bool = False, select_cfg: dict | None = None,
                   balance: dict | None = None):
    kind = model_cfg.get("type", {"classification": "logistic", "regression": "ridge", "survival": "coxnet"}[task])
    parts = []
    if gene_cols:
        # Gene columns must come first in X, in this order. They are selected by
        # position: selecting 20,000+ columns by name is quadratic in scikit-learn
        # (about 3 s per fit for 19,595 genes) and dominated the run time.
        steps = [("filter", ExpressionFilter(**(filter_cfg or {}))), ("impute", SimpleImputer(strategy="median"))]
        if select_cfg:
            from .selection import GeneSelector
            steps.append(("select", GeneSelector(**select_cfg)))
        steps.append(("scale", StandardScaler()))
        parts.append(("genes", Pipeline(steps), slice(0, len(gene_cols))))
    if numeric_covariates:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), list(numeric_covariates)))
    if categorical_covariates:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                      list(categorical_covariates)))
    if not parts:
        raise ValueError("the model has no inputs: no genes and no covariates")
    pipe = Pipeline([("pre", ColumnTransformer(parts)), ("model", make_model(task, kind, seed))])
    fixed = _translate(task, kind, model_cfg.get("params"))
    for k, v in fixed.items():
        if task == "survival" and kind == "coxnet" and k == "alphas" and v and isinstance(v[0], list):
            v = v[0]
        pipe.set_params(**{f"model__{k}": v})
    grid = _translate(task, kind, model_cfg.get("grid"))
    est = pipe
    if grid and any(len(v) > 1 for v in grid.values()):
        est = GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()},
                           cv=inner_cv(task, model_cfg.get("inner_folds", 3), seed, grouped),
                           scoring=scoring_for(task, n_classes), refit=True, n_jobs=1, error_score=np.nan)
    elif grid:
        pipe.set_params(**{f"model__{k}": v[0] for k, v in grid.items()})
    if balance and balance.get("method", "weights") == "undersample":
        if task != "classification":
            raise ValueError("balance: undersample applies to classification only")
        from .balance import UndersampleEnsemble
        est = UndersampleEnsemble(est, n_bags=balance.get("n_bags", 10), ratio=balance.get("ratio", 1.0), seed=seed)
    elif balance and balance.get("method") not in (None, "weights"):
        raise ValueError("balance.method must be 'weights' or 'undersample'")
    return est


def _is_ensemble(est) -> bool:
    return type(est).__name__ == "UndersampleEnsemble"


def fit_estimator(est, X, y, groups=None):
    if isinstance(est, GridSearchCV) or _is_ensemble(est):
        return est.fit(X, y, groups=groups)
    return est.fit(X, y)


def final_pipelines(est) -> list[Pipeline]:
    """All fitted pipelines inside an estimator (one, or one per bag)."""
    if _is_ensemble(est):
        return [p for e in est.estimators_ for p in final_pipelines(e)]
    return [est.best_estimator_ if isinstance(est, GridSearchCV) else est]


def final_pipeline(est) -> Pipeline:
    return final_pipelines(est)[0]


def best_params(est) -> dict:
    if _is_ensemble(est):
        chosen = [best_params(e) for e in est.estimators_]
        if not chosen or not chosen[0]:
            return {}
        # most frequent choice across bags
        keys = chosen[0].keys()
        out = {}
        for k in keys:
            vals = pd.Series([str(c[k]) for c in chosen]).value_counts()
            pick = vals.index[0]
            out[k] = next(c[k] for c in chosen if str(c[k]) == pick)
        return out
    if isinstance(est, GridSearchCV):
        return {k.replace("model__", ""): v for k, v in est.best_params_.items()}
    return {}


def coefficients(est) -> pd.Series | None:
    """Coefficients (linear models) or importances (forests) by input feature name."""
    if _is_ensemble(est):
        parts = [coefficients(e) for e in est.estimators_]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1).rename(parts[0].name)
    pipe = final_pipeline(est)
    model = pipe.named_steps["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    if hasattr(model, "coef_"):
        c = np.asarray(model.coef_, dtype=float)
        if c.ndim == 2 and c.shape[0] == len(names):      # coxnet: (features, alphas)
            c = c[:, -1]
        elif c.ndim == 2 and c.shape[0] == 1:             # binary logistic
            c = c[0]
        elif c.ndim == 2:                                  # multiclass: largest absolute value per feature
            c = c[np.abs(c).argmax(axis=0), np.arange(c.shape[1])]
        return pd.Series(c, index=names, name="coefficient")
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=names, name="importance")
    return None
