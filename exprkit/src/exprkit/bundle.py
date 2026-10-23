"""Saved models: write, read, apply to new samples, evaluate on new labelled samples.

A bundle is a joblib file holding the fitted pipeline plus what is needed to
use it safely: the gene list in training order, covariate names, classes,
the training data transform, training gene medians for a scale check, and
survival risk group cut points.

Security note: joblib files are pickles. Loading one can run arbitrary code,
so only load bundles you created or obtained from a source you trust.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .integrate import align_to_genes
from .samples import category_values

FORMAT_VERSION = 1


def save_bundle(path: str | Path, content: dict) -> None:
    import sklearn
    meta = {"format_version": FORMAT_VERSION, "exprkit_version": __version__,
            "sklearn_version": sklearn.__version__, "created": _dt.datetime.now().isoformat(timespec="seconds")}
    joblib.dump({**meta, **content}, path, compress=3)


def load_bundle(path: str | Path) -> dict:
    b = joblib.load(path)
    if b.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported bundle format {b.get('format_version')}")
    import sklearn
    if b.get("sklearn_version") != sklearn.__version__:
        print(f"warning: model saved with scikit-learn {b.get('sklearn_version')}, running {sklearn.__version__}")
    return b


def prepare_input(bundle: dict, expression: pd.DataFrame | None, samples: pd.DataFrame | None,
                  max_missing_fraction: float = 0.2, allow_missing_covariates: bool = False) -> tuple[pd.DataFrame, dict]:
    """Build the model input for new samples and report every compromise.

    * Genes are matched by exact symbol. The missing fraction is computed on
      the genes the fitted model actually uses (after its filter); above
      ``max_missing_fraction`` the run stops. Missing genes are filled by
      the pipeline's imputer with the training median.
    * Covariates come from ``samples``; missing covariates stop the run
      unless ``allow_missing_covariates``.
    * Scale check: compares gene medians of the new data with training
      medians on the genes the model uses.
    """
    rep: dict = {}
    genes = bundle["gene_columns"]
    norm = bundle["training"].get("normalise", "none")
    if norm == "per_sample_rank" and expression is not None and genes:
        # rank over exactly the genes the training data were ranked over
        from .transform import per_sample_rank
        full, rep["rank_genes"] = align_to_genes(expression, bundle["training"]["rank_genes"], 1.0)
        expression = per_sample_rank(full)
        rep["normalise"] = "per_sample_rank applied to the new samples"
    index = expression.index if expression is not None else samples.index
    X = pd.DataFrame(index=index)
    if genes:
        used = bundle.get("genes_used_by_model") or genes
        _, rep["genes_used_by_model"] = align_to_genes(expression, used, max_missing_fraction)
        Xg, rep["all_training_genes"] = align_to_genes(expression, genes, 1.0)
        X = Xg.astype("float64")
        med = bundle["training"].get("gene_medians")
        if med is not None:
            new_med = X[used].median(axis=0)
            ok = new_med.notna() & med.reindex(used).notna()
            shift = float((new_med[ok] - med.reindex(used)[ok]).median())
            corr = float(pd.Series(new_med[ok]).corr(med.reindex(used)[ok], method="spearman"))
            sc = {"median_shift_new_minus_training": round(shift, 3), "spearman_of_gene_medians": round(corr, 3)}
            warns = []
            if abs(shift) > 1.0:
                warns.append(f"gene medians differ from training by {shift:+.2f} on average; the new data may use "
                             "another scale or transform (training transform: "
                             f"{bundle['training'].get('transforms')})")
            if corr < 0.7:
                warns.append(f"gene medians correlate weakly with training (Spearman {corr:.2f}); a different "
                             "platform or preprocessing is likely, and predictions may not be valid")
            sc["warnings"] = warns
            rep["scale_check"] = sc
            for w in warns:
                print("WARNING:", w)
    for c in bundle["numeric_covariates"] + bundle["categorical_covariates"]:
        if samples is None or c not in samples.columns:
            if not allow_missing_covariates:
                raise KeyError(f"covariate '{c}' is required by the model but not provided")
            X[c] = np.nan
            rep.setdefault("missing_covariates", []).append(c)
        elif c in bundle["numeric_covariates"]:
            X[c] = pd.to_numeric(samples.loc[index, c], errors="coerce").to_numpy()
        else:
            X[c] = category_values(samples.loc[index, c])
    return X, rep


def predict(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    est = bundle["estimator"]
    task = bundle["task"]
    out = pd.DataFrame(index=X.index)
    if task == "classification":
        proba = est.predict_proba(X)
        for j, c in enumerate(bundle["classes"]):
            out[f"prob_{c}"] = proba[:, j]
        out["predicted_class"] = np.asarray(bundle["classes"])[proba.argmax(axis=1)]
    elif task == "survival":
        from .evaluation import assign_risk_groups
        risk = np.asarray(est.predict(X), float)
        out["risk_score"] = risk
        out["risk_group"] = assign_risk_groups(risk, bundle["risk_cutpoints"])
    else:
        out["predicted"] = np.asarray(est.predict(X), float)
    return out


def evaluate_external(bundle: dict, X: pd.DataFrame, samples: pd.DataFrame, out_dir: str | Path,
                      n_boot: int = 1000, seed: int = 0) -> dict:
    """Score a saved model on new labelled samples (external validation)."""
    from . import plots
    from .metrics import (auc_score, bootstrap_interval, classification_metrics, harrell_c,
                          regression_metrics)
    from .train import build_target
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task = bundle["task"]
    y, ok, excl = build_target(task, samples.loc[X.index], bundle["target"])
    X, y = X.loc[ok], y[ok]
    pred = predict(bundle, X)
    res: dict = {"task": task, "n_used": int(len(X)), **excl}
    if task == "classification":
        classes, pos = bundle["classes"], bundle["positive_class"]
        unknown = sorted(set(y) - set(map(str, classes)))
        if unknown:
            raise ValueError(f"labels {unknown} were not seen in training (classes {classes})")
        proba = pred[[f"prob_{c}" for c in classes]].to_numpy()
        m = classification_metrics(y, proba, classes, pos)
        res["metrics"] = m
        if len(set(y)) > 1:
            res["bootstrap_auc"] = dict(zip(["auc", "ci_low", "ci_high"], bootstrap_interval(
                lambda ii: auc_score(y[ii], proba[ii], classes, pos), len(y), n_boot, seed, strata=y)))
            if len(classes) == 2:
                plots.roc(y, [proba[:, classes.index(pos)]], pos, out / "roc_external.png", "External evaluation")
        plots.confusion(m["confusion_matrix"]["rows_true_cols_pred"], [str(c) for c in classes],
                        out / "confusion_external.png", "External evaluation")
    elif task == "survival":
        r = pred["risk_score"].to_numpy()
        res["bootstrap_c"] = dict(zip(["c", "ci_low", "ci_high"], bootstrap_interval(
            lambda ii: harrell_c(y["time"][ii], y["event"][ii], r[ii]), len(r), n_boot, seed)))
        res["kaplan_meier"] = plots.km_by_group(y["time"], y["event"], pred["risk_group"], out / "km_external.png",
                                                "External evaluation (training cut points)")
    else:
        res["metrics"] = regression_metrics(y, pred["predicted"])
        res["bootstrap_r2"] = dict(zip(["r2", "ci_low", "ci_high"], bootstrap_interval(
            lambda ii: regression_metrics(y[ii], pred["predicted"].to_numpy()[ii])["r2"], len(y), n_boot, seed)))
        plots.scatter(y, pred["predicted"], out / "observed_vs_predicted_external.png", "External evaluation")
    pred.to_csv(out / "predictions.csv")
    return res
