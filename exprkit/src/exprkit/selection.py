"""Gene selection: univariate importance, agreement across datasets, model
stability, and removal of highly correlated genes.

Two ways to use it
------------------
1. ``exprkit select-genes``: build a gene list from one or more prepared
   datasets and write it to a file (``genes.txt``) with a full table of the
   evidence for every gene. Use the list in ``train`` with
   ``genes.gene_list``, ideally on data that played no part in choosing it.

2. ``genes.select`` in a train config: the same univariate and correlation
   steps run inside every training fold (``GeneSelector``), so the evaluation
   includes the effect of choosing genes. Selecting genes on all samples and
   then cross validating on the same samples is optimistic, sometimes
   dramatically (Ambroise and McLachlan 2002, PNAS 99:6562).

Importance scores (signed z statistics, larger magnitude = stronger)
--------------------------------------------------------------------
classification, two classes   Welch t statistic, converted to a z value
classification, more classes  ANOVA F statistic, converted to a (positive) z value
survival                      Cox score test z for each gene alone (Breslow ties);
                              positive = higher expression, higher hazard
regression                    Fisher z of the Pearson correlation, times sqrt(n - 3)

Across datasets, z values are combined with Stouffer's method
(sum of z / sqrt(number of datasets)), and the fraction of datasets agreeing
in sign with the combined direction is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


# Univariate scores ------------------------------------------------------------------

def infer_task(y) -> str:
    y = np.asarray(y)
    if y.dtype.names and "event" in y.dtype.names:
        return "survival"
    if y.dtype.kind in "fc" and len(np.unique(y[~pd.isna(y)])) > 10:
        return "regression"
    return "classification"


def cox_score_z(X: np.ndarray, time, event) -> np.ndarray:
    """Score test z for each column of X in a one variable Cox model (Breslow ties).

    U = sum over events of (x_i - mean of x in the risk set);
    V = sum over events of (variance of x in the risk set); z = U / sqrt(V).
    Vectorised over genes with cumulative sums; tested against a loop.
    """
    t, e = np.asarray(time, float), np.asarray(event).astype(bool)
    X = np.asarray(X, float)
    order = np.argsort(-t, kind="stable")
    ts, es, Xs = t[order], e[order], X[order]
    s0 = np.arange(1, len(ts) + 1, dtype=float)
    s1 = np.cumsum(Xs, axis=0)
    s2 = np.cumsum(Xs * Xs, axis=0)
    # for tied times, the risk set includes everyone with the same time
    last = np.searchsorted(-ts, -ts, side="right") - 1
    s0, s1, s2 = s0[last], s1[last], s2[last]
    m = s1 / s0[:, None]
    U = (Xs[es] - m[es]).sum(axis=0)
    V = (s2[es] / s0[es][:, None] - m[es] ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = U / np.sqrt(V)
    return np.nan_to_num(z)


def univariate_z(X: np.ndarray, y, task: str | None = None, positive=None) -> np.ndarray:
    """Signed z statistic per column (unsigned for more than two classes)."""
    X = np.asarray(X, float)
    task = task or infer_task(y)
    n = X.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        if task == "survival":
            return cox_score_z(X, y["time"], y["event"])
        if task == "regression":
            yv = np.asarray(y, float)
            xc = X - X.mean(axis=0)
            yc = yv - yv.mean()
            r = (xc * yc[:, None]).sum(axis=0) / np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
            r = np.clip(np.nan_to_num(r), -0.999999, 0.999999)
            return np.arctanh(r) * np.sqrt(max(n - 3, 1))
        y = np.asarray(y).astype(str)
        classes = sorted(set(y))
        if len(classes) == 2:
            pos = str(positive) if positive is not None else classes[-1]
            a, b = X[y == pos], X[y != pos]
            t, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
            na, nb = len(a), len(b)
            va, vb = a.var(axis=0, ddof=1) / na, b.var(axis=0, ddof=1) / nb
            df = (va + vb) ** 2 / (va ** 2 / max(na - 1, 1) + vb ** 2 / max(nb - 1, 1))
            p_one = stats.t.sf(np.abs(t), np.nan_to_num(df, nan=1.0))
            return np.nan_to_num(np.sign(t) * stats.norm.isf(np.clip(p_one, 1e-300, 0.5)))
        from sklearn.feature_selection import f_classif
        F, p = f_classif(np.nan_to_num(X), y)
        return np.nan_to_num(stats.norm.isf(np.clip(p, 1e-300, 0.5)))


# Correlation pruning --------------------------------------------------------------------

def prune_correlated(X: np.ndarray, ranked: np.ndarray, max_correlation: float, n_keep: int | None) -> np.ndarray:
    """Walk down ``ranked`` (column indices, best first) and keep a column only if
    its absolute Pearson correlation with every column already kept is at most
    ``max_correlation``. Stops after ``n_keep`` columns."""
    X = np.asarray(X, float)
    Z = X - np.nanmean(X, axis=0)
    sd = np.sqrt(np.nanmean(Z ** 2, axis=0))
    sd[sd == 0] = np.inf
    Z = np.nan_to_num(Z / sd)
    kept: list[int] = []
    n = X.shape[0]
    for j in ranked:
        if kept and max_correlation < 1:
            c = np.abs(Z[:, kept].T @ Z[:, j]) / n
            if c.max() > max_correlation:
                continue
        kept.append(int(j))
        if n_keep is not None and len(kept) >= n_keep:
            break
    return np.asarray(kept, dtype=int)


class GeneSelector(BaseEstimator, TransformerMixin):
    """Keep the ``n_genes`` most associated genes, skipping any gene correlated
    above ``max_correlation`` with a better one. Fitted on training rows only
    when used inside a pipeline, so cross validation stays honest."""

    def __init__(self, n_genes: int = 100, max_correlation: float = 0.9, positive=None):
        self.n_genes = n_genes
        self.max_correlation = max_correlation
        self.positive = positive

    def fit(self, X, y):
        names = np.asarray(X.columns.astype(str), dtype=object) if isinstance(X, pd.DataFrame) \
            else np.asarray([f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object)
        V = np.asarray(X, float)
        self.feature_names_in_ = names
        self.n_features_in_ = V.shape[1]
        z = univariate_z(V, y, positive=self.positive)
        self.scores_ = z
        ranked = np.argsort(-np.abs(z), kind="stable")
        keep = prune_correlated(V, ranked, self.max_correlation, self.n_genes)
        self.support_ = np.zeros(V.shape[1], dtype=bool)
        self.support_[keep] = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.support_]
        return np.asarray(X)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features, dtype=object) if input_features is not None else self.feature_names_in_
        return names[self.support_]


# Building a gene list from several datasets -------------------------------------------

def _standardise(X: pd.DataFrame) -> pd.DataFrame:
    sd = X.std(axis=0, ddof=0).replace(0, np.nan)
    return (X - X.mean(axis=0)) / sd


def stability_selection(X: np.ndarray, y, task: str, n_boot: int = 50, seed: int = 0,
                        strength: float | None = None, positive=None) -> np.ndarray:
    """Fraction of half samples in which an L1 penalised model gives each gene a
    non zero coefficient (Meinshausen and Buhlmann 2010, JRSS B 72:417).

    classification: L1 logistic (two classes) or L1 multinomial; survival: lasso
    Cox (coxnet); regression: lasso. ``strength`` is C for logistic, alpha otherwise.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    counts = np.zeros(p)
    Xn = np.nan_to_num((X - np.nanmean(X, 0)) / np.where(np.nanstd(X, 0) == 0, 1, np.nanstd(X, 0)))
    for _ in range(n_boot):
        if task == "classification":
            yy = np.asarray(y).astype(str)
            idx = np.concatenate([rng.choice(np.flatnonzero(yy == c), max(1, (yy == c).sum() // 2), replace=False)
                                  for c in np.unique(yy)])
        else:
            idx = rng.choice(n, n // 2, replace=False)
        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(l1_ratio=1.0, solver="saga", C=strength or 0.05, class_weight="balanced",
                                   max_iter=5000).fit(Xn[idx], np.asarray(y)[idx])
            coef = np.abs(m.coef_).max(axis=0)
        elif task == "survival":
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[strength or 0.05], max_iter=100000).fit(Xn[idx], y[idx])
            coef = np.abs(m.coef_[:, -1])
        else:
            from sklearn.linear_model import Lasso
            m = Lasso(alpha=strength or 0.05, max_iter=20000).fit(Xn[idx], np.asarray(y, float)[idx])
            coef = np.abs(m.coef_)
        counts += coef > 1e-10
    return counts / n_boot


def select_genes(datasets: dict[str, tuple[pd.DataFrame, object]], task: str, n_genes: int = 100,
                 min_datasets: int | None = None, min_sign_agreement: float = 1.0,
                 max_correlation: float = 0.9, max_mode_fraction: float = 0.95,
                 stability: dict | None = None, positive=None, seed: int = 0) -> pd.DataFrame:
    """Rank genes across datasets and return the evidence table.

    ``datasets`` maps a name to (samples x genes expression, target). Steps:

    1. Occurrence: a gene counts as present in a dataset when it has values for
       at least half of the samples and is not quasi constant there. Genes
       present in fewer than ``min_datasets`` datasets (default: all) are dropped.
    2. Importance per dataset: signed z statistic (see module docstring).
    3. Agreement: Stouffer combined z, and the fraction of datasets whose sign
       matches it; genes below ``min_sign_agreement`` are dropped.
    4. Optional stability selection on the pooled data (each dataset
       standardised per gene first), restricted to the best ``stability.top``
       genes by combined z for speed.
    5. Ranking by stability (if computed) then |combined z|, and greedy removal
       of genes correlated above ``max_correlation`` with a better ranked gene,
       measured on the pooled per dataset standardised data.
    """
    from .filtering import mode_fraction
    names = list(datasets)
    all_genes = sorted(set().union(*[set(X.columns) for X, _ in datasets.values()]))
    table = pd.DataFrame(index=pd.Index(all_genes, name="gene"))
    present = pd.DataFrame(False, index=table.index, columns=names)
    for d, (X, y) in datasets.items():
        Xa = X.reindex(columns=all_genes)
        ok_values = Xa.notna().mean(axis=0) >= 0.5
        mf = pd.Series(mode_fraction(Xa.to_numpy(float)), index=all_genes)
        present[d] = ok_values & (mf < max_mode_fraction)
        z = pd.Series(np.nan, index=all_genes)
        cols = present.index[present[d]]
        Xd = Xa[cols].astype(float)
        Xd = Xd.fillna(Xd.median())
        z[cols] = univariate_z(Xd.to_numpy(), y, task, positive)
        table[f"z_{d}"] = z
    table["n_datasets_present"] = present.sum(axis=1)
    need = len(names) if min_datasets is None else min_datasets
    zcols = [f"z_{d}" for d in names]
    zmat = table[zcols]
    table["combined_z"] = zmat.sum(axis=1, min_count=1) / np.sqrt(zmat.notna().sum(axis=1).replace(0, np.nan))
    sign = np.sign(table["combined_z"])
    table["sign_agreement"] = (np.sign(zmat).eq(sign, axis=0) & zmat.notna()).sum(axis=1) / zmat.notna().sum(axis=1)
    if task == "classification" and any(len(set(np.asarray(y).astype(str))) > 2 for _, y in datasets.values()):
        table["sign_agreement"] = 1.0          # more than two classes: z values are unsigned
    eligible = (table["n_datasets_present"] >= need) & (table["sign_agreement"] >= min_sign_agreement) \
        & table["combined_z"].notna()
    table["eligible"] = eligible

    # pooled, per dataset standardised data for stability and correlation
    cand = table.index[eligible]
    pooled = pd.concat([_standardise(X.reindex(columns=cand).astype(float)) for X, _ in datasets.values()])
    pooled = pooled.fillna(0.0)
    ys = [y for _, y in datasets.values()]
    y_pooled = np.concatenate(ys) if ys[0] is not None else None

    order_key = table.loc[cand, "combined_z"].abs()
    if stability and stability.get("n_boot", 0) > 0 and len(cand):
        top = order_key.sort_values(ascending=False).index[: stability.get("top", 500)]
        freq = stability_selection(pooled[top].to_numpy(), y_pooled, task, stability["n_boot"], seed,
                                   stability.get("strength"), positive)
        table["stability"] = np.nan
        table.loc[top, "stability"] = freq
        st = table.loc[cand, "stability"].fillna(-1)
        ranked_genes = sorted(cand, key=lambda g: (-st[g], -order_key[g]))
    else:
        ranked_genes = list(order_key.sort_values(ascending=False).index)
    idx = {g: i for i, g in enumerate(cand)}
    keep = prune_correlated(pooled[cand].to_numpy(), np.array([idx[g] for g in ranked_genes]),
                            max_correlation, n_genes)
    kept_genes = [cand[i] for i in keep]
    table["rank"] = np.nan
    table.loc[ranked_genes, "rank"] = np.arange(1, len(ranked_genes) + 1)
    table["selected"] = table.index.isin(kept_genes)
    table["removed_as_correlated"] = table.index.isin(ranked_genes) & ~table["selected"] & \
        (table["rank"] <= (max(table.loc[kept_genes, "rank"]) if kept_genes else 0))
    return table.sort_values(["selected", "rank"], ascending=[False, True])


def run_select_genes(cfg: dict, base=".") -> pd.DataFrame:
    """``exprkit select-genes``: build a gene list from prepared datasets (see module docstring)."""
    from pathlib import Path

    import yaml

    from .io import load_dataset, write_json
    from .train import build_target, select_rows

    def res(p):
        p = Path(p)
        return p if p.is_absolute() else Path(base) / p

    out = res(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    task = cfg["task"]
    data = cfg["data"] if isinstance(cfg["data"], list) else [cfg["data"]]
    datasets: dict[str, tuple[pd.DataFrame, object]] = {}
    counts = {}
    for d in data:
        X, S, _ = load_dataset(res(d))
        m = select_rows(S, cfg.get("where"))
        y_all, ok, _ = build_target(task, S, cfg.get("target", {}))
        rows = m & ok
        X, S, y = X.loc[rows], S.loc[rows], y_all[rows]
        gcol = cfg.get("group_column")
        if gcol and cfg.get("one_per_group", False):
            first = ~S[gcol].astype(str).duplicated().to_numpy()
            X, S, y = X.loc[first], S.loc[first], y[first]
        if cfg.get("gene_list"):
            wanted = [g.strip().upper() for g in res(cfg["gene_list"]).read_text().split() if g.strip()]
            X = X.reindex(columns=[g for g in wanted if g in set(X.columns)])
        split = cfg.get("split_by")
        parts = [(Path(d).name, slice(None))] if not split else \
            [(str(v), (S[split].astype(str) == str(v)).to_numpy()) for v in sorted(S[split].astype(str).unique())]
        for name, msk in parts:
            Xi, yi = X.loc[msk], y[msk]
            if task == "classification" and len(set(np.asarray(yi).astype(str))) < 2:
                print(f"skipping {name}: only one class, so no association can be measured there")
                continue
            if task == "survival" and np.asarray(yi["event"]).sum() < 5:
                print(f"skipping {name}: fewer than 5 events")
                continue
            datasets[name] = (Xi, yi)
            counts[name] = int(len(yi))
    if not datasets:
        raise ValueError("no dataset has enough information for selection")
    print(f"Selecting genes from {len(datasets)} dataset(s): {counts}")
    table = select_genes(datasets, task, cfg.get("n_genes", 100), cfg.get("min_datasets"),
                         cfg.get("min_sign_agreement", 1.0), cfg.get("max_correlation", 0.9),
                         cfg.get("max_mode_fraction", 0.95), cfg.get("stability"),
                         (cfg.get("target") or {}).get("positive_class"), int(cfg.get("seed", 42)))
    sel = table.index[table["selected"]].tolist()
    (out / "genes.txt").write_text("\n".join(sel) + "\n")
    table.to_csv(out / "gene_selection.csv")
    with open(out / "config_used.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    write_json({"datasets": counts, "n_candidates": int(len(table)), "n_eligible": int(table["eligible"].sum()),
                "n_selected": len(sel), "n_removed_as_correlated": int(table["removed_as_correlated"].sum()),
                "note": "This list was chosen using these samples. Scores of a model trained and tested on the "
                        "same samples with this list are optimistic; evaluate on independent data, or use "
                        "genes.select in train to repeat the selection inside each training fold."},
               out / "select_report.json")
    print(f"{len(sel)} genes selected ({int(table['removed_as_correlated'].sum())} skipped as correlated) -> {out / 'genes.txt'}")
    return table
