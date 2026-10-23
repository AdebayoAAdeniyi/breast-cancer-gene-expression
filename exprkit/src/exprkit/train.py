"""``exprkit train``: evaluate a model honestly, then fit it on all rows and save it.

Outputs in the ``output`` folder
--------------------------------
summary.json            data counts, evaluation summary with intervals, baseline comparison,
                        permutation test, design checks, final model settings
folds.csv               score of every outer fold (AUC, Harrell's C or R2)
per_repeat.csv          metrics of the pooled out of fold predictions of each repeat
oof_predictions.csv     out of fold predictions (long format)
feature_stability.csv   how often each feature was among the top features across fits
coefficients.csv        coefficients (or importances) of the final model
confounding.csv         target by batch table (if checks.batch_column is set)
model.joblib            the final model bundle for ``exprkit predict`` and ``exprkit evaluate``
*.png                   ROC, confusion matrix, Kaplan Meier or scatter plot
config_used.yaml        the exact configuration
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import plots
from .bundle import save_bundle
from .evaluation import confounding_table, cross_validate, make_splits, permutation_test
from .io import load_dataset, write_json
from .metrics import (bootstrap_interval, classification_metrics, corrected_resampled_t, harrell_c,
                      paired_bootstrap, regression_metrics)
from .samples import category_values
from .models import best_params, build_pipeline, coefficients, final_pipeline, final_pipelines, fit_estimator


def _resolve(p, base):
    p = Path(p)
    return p if p.is_absolute() else Path(base) / p


def select_rows(samples: pd.DataFrame, where: dict | None) -> np.ndarray:
    m = np.ones(len(samples), dtype=bool)
    for col, vals in (where or {}).items():
        if col not in samples.columns:
            raise KeyError(f"where: column '{col}' not in samples.csv. Available: {list(samples.columns)}")
        vals = vals if isinstance(vals, list) else [vals]
        m &= samples[col].astype(str).isin([str(v) for v in vals]).to_numpy()
    return m


def build_target(task: str, samples: pd.DataFrame, target: dict) -> tuple[object, np.ndarray, dict]:
    """Return y, a mask of usable rows, and a report of exclusions."""
    rep = {}
    if task == "survival":
        tcol, ecol = target.get("time", "time"), target.get("event", "event")
        for c in (tcol, ecol):
            if c not in samples.columns:
                raise KeyError(f"target column '{c}' not in samples.csv")
        t = pd.to_numeric(samples[tcol], errors="coerce").to_numpy(float)
        e = pd.to_numeric(samples[ecol], errors="coerce").to_numpy(float)
        ok = ~np.isnan(t) & (t > 0) & np.isin(e, [0, 1])
        rep = {"missing_or_nonpositive_time": int((np.isnan(t) | (t <= 0)).sum()),
               "missing_or_invalid_event": int((~np.isin(e, [0, 1])).sum())}
        y = np.array(list(zip(e.astype(bool), t)), dtype=[("event", bool), ("time", float)])
        return y, ok, rep
    col = target["column"]
    if col not in samples.columns:
        raise KeyError(f"target column '{col}' not in samples.csv. Available: {list(samples.columns)}")
    if task == "classification":
        v = samples[col]
        ok = v.notna().to_numpy() & (v.astype(str).str.strip() != "").to_numpy()
        rep = {"missing_target": int((~ok).sum())}
        return v.astype(str).to_numpy(), ok, rep
    v = pd.to_numeric(samples[col], errors="coerce").to_numpy(float)
    ok = ~np.isnan(v)
    rep = {"missing_target": int((~ok).sum())}
    return v, ok, rep


def _summ(df: pd.DataFrame) -> dict:
    num = df.select_dtypes("number").drop(columns=["repeat"], errors="ignore")
    return {c: {"mean": float(num[c].mean()), "p2.5": float(num[c].quantile(0.025)),
                "p97.5": float(num[c].quantile(0.975))} for c in num.columns}


def train(cfg: dict, base: str | Path = ".") -> dict:
    seed = int(cfg.get("seed", 42))
    task = cfg["task"]
    if task not in ("classification", "survival", "regression"):
        raise ValueError("task must be classification, survival or regression")
    out = _resolve(cfg["output"], base)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_used.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    data_dir = _resolve(cfg["data"], base)
    expr, samples, dreport = load_dataset(data_dir)
    summary: dict = {"task": task, "data": str(data_dir)}

    # Rows -----------------------------------------------------------------------
    m = select_rows(samples, cfg.get("where"))
    y_all, ok, excl = build_target(task, samples, cfg.get("target", {}))
    rows = m & ok
    summary["rows"] = {"in_dataset": int(len(samples)), "after_where": int(m.sum()), **excl,
                       "used": int(rows.sum())}
    S = samples.loc[rows]
    y = y_all[rows]
    gcol = cfg.get("group_column")
    groups = S[gcol].astype(str).to_numpy() if gcol else None
    if gcol and cfg.get("one_per_group", False):
        first = ~pd.Series(groups).duplicated().to_numpy()
        summary["rows"]["dropped_extra_profiles_per_group"] = int((~first).sum())
        S, y, groups = S.loc[first], y[first], groups[first]
        summary["rows"]["used"] = int(len(S))

    # Features -------------------------------------------------------------------
    gcfg = cfg.get("genes", {}) or {}
    gene_cols: list[str] = []
    if gcfg.get("use", True):
        gene_cols = list(expr.columns)
        if gcfg.get("gene_list"):
            wanted = [g.strip().upper() for g in _resolve(gcfg["gene_list"], base).read_text().split() if g.strip()]
            present = [g for g in wanted if g in set(gene_cols)]
            summary["gene_list"] = {"requested": len(wanted), "found": len(present),
                                    "missing": sorted(set(wanted) - set(present))[:50]}
            gene_cols = present
    cov = cfg.get("covariates", {}) or {}
    num_cov, cat_cov = list(cov.get("numeric", []) or []), list(cov.get("categorical", []) or [])
    for c in num_cov + cat_cov:
        if c not in S.columns:
            raise KeyError(f"covariate '{c}' not in samples.csv")
    X = expr.loc[S.index, gene_cols].astype("float64")
    for c in num_cov:
        X[c] = pd.to_numeric(S[c], errors="coerce").to_numpy()
    for c in cat_cov:
        X[c] = category_values(S[c])
    assert list(X.columns[:len(gene_cols)]) == gene_cols  # models select genes by position
    summary["features"] = {"genes": len(gene_cols), "numeric_covariates": num_cov,
                           "categorical_covariates": cat_cov}

    classes, positive = None, None
    if task == "classification":
        classes = sorted(pd.unique(y).tolist())
        if len(classes) < 2:
            raise ValueError(f"only one class in the training rows: {classes}")
        positive = str((cfg.get("target") or {}).get("positive_class", classes[-1]))
        if len(classes) == 2 and positive not in classes:
            raise ValueError(f"positive_class '{positive}' is not one of {classes}")
        summary["class_counts"] = pd.Series(y).value_counts().to_dict()
    elif task == "survival":
        summary["events"] = {"n_events": int(y["event"].sum()), "n_censored": int((~y["event"]).sum()),
                             "median_time": float(np.median(y["time"]))}

    # Design checks --------------------------------------------------------------
    bcol = (cfg.get("checks") or {}).get("batch_column")
    if bcol and bcol in S.columns and task == "classification":
        tab, warns = confounding_table(pd.Series(y, index=S.index), S[bcol])
        tab.to_csv(out / "confounding.csv")
        summary["confounding"] = {"table": tab.to_dict(), "warnings": warns}
        for w in warns:
            print("WARNING:", w)

    # Estimators -----------------------------------------------------------------
    mcfg = cfg.get("model", {}) or {}
    fcfg = gcfg.get("filter", {}) or {}
    n_classes = len(classes) if classes else 2
    grouped = groups is not None

    scfg = gcfg.get("select")
    if scfg and task == "classification" and n_classes == 2:
        scfg = {**scfg, "positive": positive}

    bal = cfg.get("balance")

    def build():
        return build_pipeline(task, mcfg, gene_cols, num_cov, cat_cov, fcfg, seed, n_classes, grouped, scfg, bal)

    ecfg = cfg.get("evaluation", {}) or {}
    split_groups = None
    if ecfg.get("type") == "leave_one_group_out":
        col = ecfg.get("column", "dataset")
        if col not in S.columns:
            raise KeyError(f"evaluation.column '{col}' not in samples.csv")
        split_groups = S[col].astype(str).to_numpy()
    splits = make_splits(task, y, groups, ecfg, seed, split_groups)
    n_tr = float(np.mean([len(tr) for *_, tr, _ in splits]))
    n_te = float(np.mean([len(te) for *_, te in splits]))
    print(f"Evaluating {mcfg.get('type', 'default')} on {len(S)} samples, {len(gene_cols)} genes, "
          f"{len(splits)} fits ({ecfg.get('type', 'repeated_cv')})")
    cv = cross_validate(task, build, X, y, groups, splits, classes, positive)
    if split_groups is not None:
        names = sorted(set(split_groups))
        cv["folds"].insert(2, "held_out", [names[f] for f in cv["folds"]["fold"]])
    cv["folds"].to_csv(out / "folds.csv", index=False)
    is_cv = ecfg.get("type", "repeated_cv") == "repeated_cv"
    n_boot = int(ecfg.get("bootstrap", 1000))
    ev: dict = {"scheme": ecfg, "score_name": {"classification": "roc_auc", "survival": "harrell_c",
                                                 "regression": "r2"}[task]}
    if split_groups is not None:
        ev["per_held_out_group"] = cv["folds"].drop(columns=["repeat", "fold"]).set_index("held_out") \
            .to_dict(orient="index")
    if is_cv:
        ev["fold_score_corrected_interval"] = corrected_resampled_t(
            cv["folds"]["score"], n_tr, n_te, null={"classification": 0.5, "survival": 0.5, "regression": 0.0}[task],
            bounds=(0.0, 1.0) if task != "regression" else None, alternative="greater")
        ev["fold_score_corrected_interval"]["p_value_note"] = ("one sided, better than chance "
                                                               "(0.5 for AUC and C, 0 for R2)")

    # Per repeat metrics, bootstrap and plots -------------------------------------
    per_rep, oof_rows = [], []
    for rep, pred in cv["preds"].items():
        filled = ~np.isnan(pred if pred.ndim == 1 else pred[:, 0])
        idx = np.flatnonzero(filled)
        if task == "classification":
            mtr = classification_metrics(y[idx], pred[idx], classes, positive)
            per_rep.append({"repeat": rep, **{k: v for k, v in mtr.items() if not isinstance(v, dict)}})
            for i in idx:
                oof_rows.append({"sample": S.index[i], "repeat": rep, "true": y[i],
                                 **{f"prob_{c}": pred[i, j] for j, c in enumerate(classes)}})
        elif task == "survival":
            fr = cv["folds"][cv["folds"]["repeat"] == rep]
            per_rep.append({"repeat": rep, "harrell_c_mean_of_folds": float(fr["score"].mean())})
            for i in idx:
                oof_rows.append({"sample": S.index[i], "repeat": rep, "time": y["time"][i], "event": int(y["event"][i]),
                                 "risk_score": pred[i], "risk_group": cv["risk_groups"][rep][i]})
        else:
            per_rep.append({"repeat": rep, **regression_metrics(y[idx], pred[idx])})
            for i in idx:
                oof_rows.append({"sample": S.index[i], "repeat": rep, "true": y[i], "predicted": pred[i]})
    per_rep = pd.DataFrame(per_rep)
    per_rep.to_csv(out / "per_repeat.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "oof_predictions.csv", index=False)
    ev["per_repeat_summary"] = _summ(per_rep)

    p0 = cv["preds"][0]
    idx0 = np.flatnonzero(~np.isnan(p0 if p0.ndim == 1 else p0[:, 0]))
    if task == "classification":
        from .metrics import auc_score
        ev["bootstrap_auc_repeat0"] = dict(zip(["auc", "ci_low", "ci_high"], bootstrap_interval(
            lambda ii: auc_score(y[idx0][ii], p0[idx0][ii], classes, positive), len(idx0), n_boot, seed,
            strata=y[idx0])))
        if n_classes == 2:
            k = classes.index(positive)
            plots.roc(y, [p[:, k] for p in cv["preds"].values()], positive, out / "roc.png",
                      f"Out of fold ROC, {len(cv['preds'])} repeat(s)")
        mtr0 = classification_metrics(y[idx0], p0[idx0], classes, positive)
        plots.confusion(mtr0["confusion_matrix"]["rows_true_cols_pred"], [str(c) for c in classes],
                        out / "confusion_repeat0.png", "Out of fold predictions, repeat 0")
        ev["confusion_matrix_repeat0"] = mtr0["confusion_matrix"]
        from .metrics import exact_proportion_ci
        cm = np.asarray(mtr0["confusion_matrix"]["rows_true_cols_pred"])
        ev["recall_exact_ci_repeat0"] = {
            str(c): {"correct": int(cm[i, i]), "n": int(cm[i].sum()),
                     "recall": float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan"),
                     **dict(zip(["ci_low", "ci_high"], exact_proportion_ci(cm[i, i], cm[i].sum())))}
            for i, c in enumerate(classes)}
    elif task == "survival":
        tt, ee = y["time"][idx0], y["event"][idx0]
        if not is_cv:
            ev["bootstrap_c_test"] = dict(zip(["c", "ci_low", "ci_high"], bootstrap_interval(
                lambda ii: harrell_c(tt[ii], ee[ii], p0[idx0][ii]), len(idx0), n_boot, seed)))
        ev["kaplan_meier_repeat0"] = plots.km_by_group(
            tt, ee, cv["risk_groups"][0][idx0], out / "km_risk_groups.png",
            "Out of fold risk groups (cut points from each training fold)")
    else:
        ev["bootstrap_r2_repeat0"] = dict(zip(["r2", "ci_low", "ci_high"], bootstrap_interval(
            lambda ii: regression_metrics(y[idx0][ii], p0[idx0][ii])["r2"], len(idx0), n_boot, seed)))
        plots.scatter(y[idx0], p0[idx0], out / "observed_vs_predicted.png", "Out of fold predictions, repeat 0")
    summary["evaluation"] = ev
    if cv["stability"] is not None:
        cv["stability"].to_csv(out / "feature_stability.csv")

    # Baseline: covariates only ------------------------------------------------------
    if (num_cov or cat_cov) and gene_cols and cfg.get("compare_with_covariates_only", True):
        def build_base():
            return build_pipeline(task, {"type": {"classification": "logistic", "regression": "ridge",
                                                   "survival": "cox"}[task]},
                                  [], num_cov, cat_cov, None, seed, n_classes, grouped)
        print("Evaluating the covariates only baseline on the same splits")
        base_cv = cross_validate(task, build_base, X, y, groups, splits, classes, positive)
        d = cv["folds"]["score"].to_numpy() - base_cv["folds"]["score"].to_numpy()
        comp = {"baseline_mean_fold_score": float(base_cv["folds"]["score"].mean()),
                "model_mean_fold_score": float(cv["folds"]["score"].mean())}
        if is_cv:
            comp["difference_model_minus_baseline"] = corrected_resampled_t(d, n_tr, n_te)
        else:
            b0 = base_cv["preds"][0]
            if task == "survival":
                fa = lambda ii: harrell_c(y["time"][idx0][ii], y["event"][idx0][ii], b0[idx0][ii])
                fb = lambda ii: harrell_c(y["time"][idx0][ii], y["event"][idx0][ii], p0[idx0][ii])
                strata = None
            elif task == "classification":
                from .metrics import auc_score
                fa = lambda ii: auc_score(y[idx0][ii], b0[idx0][ii], classes, positive)
                fb = lambda ii: auc_score(y[idx0][ii], p0[idx0][ii], classes, positive)
                strata = y[idx0]
            else:
                fa = lambda ii: regression_metrics(y[idx0][ii], b0[idx0][ii])["r2"]
                fb = lambda ii: regression_metrics(y[idx0][ii], p0[idx0][ii])["r2"]
                strata = None
            comp["difference_model_minus_baseline"] = paired_bootstrap(fa, fb, len(idx0), n_boot, seed, strata)
        summary["covariates_only_baseline"] = comp

    # Final model on all rows -------------------------------------------------------
    print("Fitting the final model on all rows")
    final = build()
    fit_estimator(final, X, y, groups)
    params = best_params(final)
    coef = coefficients(final)
    if coef is not None:
        coef.to_frame().assign(abs=lambda d: d.iloc[:, 0].abs()).sort_values("abs", ascending=False) \
            .drop(columns="abs").to_csv(out / "coefficients.csv")
    pipe = final_pipeline(final)
    model_obj = final if type(final).__name__ == "UndersampleEnsemble" else pipe
    used_genes = []
    if gene_cols:
        seen = set()
        for pp in final_pipelines(final):            # union over bags when balancing by undersampling
            gpipe = pp.named_steps["pre"].named_transformers_["genes"]
            g = list(gpipe.named_steps["filter"].get_feature_names_out())
            if "select" in gpipe.named_steps:
                g = list(gpipe.named_steps["select"].get_feature_names_out(g))
            used_genes += [x for x in g if x not in seen]
            seen.update(g)
    if bal and bal.get("method") == "undersample":
        summary["balance"] = {"method": "undersample", "n_bags": bal.get("n_bags", 10),
                              "rows_per_class_in_each_bag": {str(k): v for k, v in final.bag_sizes_.items()}}
    summary["final_model"] = {"type": mcfg.get("type"), "best_params": {k: str(v) for k, v in params.items()},
                              "genes_after_filter": len(used_genes),
                              "nonzero_coefficients": int((coef != 0).sum()) if coef is not None else None,
                              "note": "Fitted on all rows used for evaluation. The evaluation above estimates the "
                                      "performance of this procedure, not of this exact fitted object."}
    risk_cut = None
    if task == "survival":
        from .evaluation import risk_cutpoints
        risk_cut = risk_cutpoints(np.asarray(pipe.predict(X), float))

    # Permutation test ----------------------------------------------------------------
    n_perm = int(ecfg.get("permutations", 0))
    if n_perm > 0:
        fixed_cfg = {**mcfg, "grid": None, "params": {**(mcfg.get("params") or {}), **params}}

        def build_fixed():
            return build_pipeline(task, fixed_cfg, gene_cols, num_cov, cat_cov, fcfg, seed, n_classes, False, scfg, bal)
        rep0 = [s for s in splits if s[0] == 0]
        print(f"Permutation test: {n_perm} permutations x {len(rep0)} fits")
        summary["permutation_test"] = permutation_test(task, build_fixed, X, y, groups, rep0, classes, positive,
                                                       n_perm, seed)

    train_medians = expr.loc[S.index, gene_cols].median(axis=0).astype("float32") if gene_cols else None
    save_bundle(out / "model.joblib", {
        "task": task, "estimator": model_obj, "target": cfg.get("target", {}), "classes": classes,
        "positive_class": positive, "gene_columns": gene_cols, "genes_used_by_model": used_genes,
        "numeric_covariates": num_cov, "categorical_covariates": cat_cov, "risk_cutpoints": risk_cut,
        "training": {"data": str(data_dir), "n_samples": int(len(S)),
                     "transforms": dreport.get("transforms"), "gene_medians": train_medians,
                     "normalise": dreport.get("normalise", "none"),
                     "rank_genes": list(expr.columns) if dreport.get("normalise") == "per_sample_rank" else None,
                     "value_range": [float(np.nanmin(X[gene_cols].to_numpy())), float(np.nanmax(X[gene_cols].to_numpy()))]
                     if gene_cols else None},
        "config": cfg})
    write_json(summary, out / "summary.json")
    _print_summary(task, summary)
    print(f"Results -> {out}")
    return summary


def _print_summary(task, s):
    ev = s["evaluation"]
    for g, r in (ev.get("per_held_out_group") or {}).items():
        extra = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                          if k.startswith(("recall_", "n_")) and k != "n_train")
        print(f"held out {g}: score {r['score']:.3f}  {extra}")
    if "fold_score_corrected_interval" in ev:
        c = ev["fold_score_corrected_interval"]
        print(f"{ev['score_name']}: {c['mean']:.3f} (95% corrected interval {c['ci_low']:.3f} to {c['ci_high']:.3f})")
    for k in ("bootstrap_auc_repeat0", "bootstrap_c_test", "bootstrap_r2_repeat0"):
        if k in ev:
            v = list(ev[k].values())
            print(f"{k}: {v[0]:.3f} ({v[1]:.3f} to {v[2]:.3f})")
    if "covariates_only_baseline" in s:
        d = s["covariates_only_baseline"]["difference_model_minus_baseline"]
        key = "mean" if "mean" in d else "difference"
        print(f"model minus covariates only: {d[key]:+.3f} ({d['ci_low']:+.3f} to {d['ci_high']:+.3f})")
    if "permutation_test" in s:
        p = s["permutation_test"]
        print(f"permutation: observed {p['observed_score_fixed_params']:.3f}, permuted mean {p['permuted_mean']:.3f}, "
              f"p = {p['p_value']:.4f}")
