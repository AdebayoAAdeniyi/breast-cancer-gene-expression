"""Gene selection: score statistics, correlation pruning, cross dataset agreement,
and the difference between selecting inside and outside cross validation."""
import numpy as np
import pandas as pd
import pytest

from exprkit.selection import GeneSelector, cox_score_z, prune_correlated, select_genes, univariate_z
from exprkit.simulate import simulate_expression, simulate_survival


def _cox_score_loop(x, t, e):
    U = V = 0.0
    for i in np.flatnonzero(e == 1):
        r = t >= t[i]
        m = x[r].mean()
        U += x[i] - m
        V += ((x[r] - m) ** 2).mean()
    return U / np.sqrt(V)


def test_cox_score_matches_loop_with_ties():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    t = rng.integers(1, 30, 120).astype(float)       # many ties
    e = rng.integers(0, 2, 120)
    z = cox_score_z(X, t, e)
    for j in range(4):
        assert z[j] == pytest.approx(_cox_score_loop(X[:, j], t, e), rel=1e-9)


def test_cox_score_direction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 2))
    t, e, _ = simulate_survival(X, np.array([0.8, 0.0]), seed=2)
    z = cox_score_z(X, t, e)
    assert z[0] > 5 and abs(z[1]) < 3


def test_welch_z_sign_and_size():
    X, y = simulate_expression(40, 40, 50, n_signal=5, effect=1.5, seed=3)
    z = univariate_z(X.to_numpy(), np.where(y == 1, "tumour", "normal"), "classification", "tumour")
    assert (z[:5] > 3).all() and np.abs(z[5:]).mean() < 1.5


def test_correlation_pruning_skips_near_copies():
    rng = np.random.default_rng(4)
    a = rng.normal(size=200)
    X = np.c_[a, a + rng.normal(0, 0.05, 200), rng.normal(size=200)]
    keep = prune_correlated(X, np.array([0, 1, 2]), 0.8, None)
    assert keep.tolist() == [0, 2]


def test_consensus_prefers_genes_consistent_across_datasets():
    rng = np.random.default_rng(5)
    data = {}
    for d in range(3):
        X, y = simulate_expression(60, 60, 60, n_signal=5, effect=1.2, seed=10 + d)
        # gene 10 is strongly associated in one dataset only, opposite sign in the others
        X.iloc[:, 10] += np.where(y == 1, 2.0 if d == 0 else -0.6, 0)
        data[f"d{d}"] = (X, np.where(y == 1, "tumour", "normal"))
    tab = select_genes(data, "classification", n_genes=5, max_correlation=0.95, positive="tumour")
    chosen = set(tab.index[tab.selected])
    assert chosen == {f"G{i:05d}" for i in range(5)}
    assert not tab.loc["G00010", "eligible"]          # direction disagrees across datasets


def test_selection_outside_cv_is_optimistic():
    """No real signal: choosing genes on all samples first inflates the CV AUC,
    choosing inside each fold does not (the Ambroise and McLachlan effect)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    X, y = simulate_expression(30, 30, 2000, n_signal=0, seed=7)
    yl = np.where(y == 1, "a", "b")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    sel = GeneSelector(n_genes=20, max_correlation=1.0).fit(X, yl)
    outside = cross_val_score(LogisticRegression(max_iter=2000), X.loc[:, sel.support_], yl, cv=cv,
                              scoring="roc_auc").mean()
    inside = cross_val_score(make_pipeline(GeneSelector(n_genes=20, max_correlation=1.0),
                                           LogisticRegression(max_iter=2000)), X, yl, cv=cv,
                             scoring="roc_auc").mean()
    assert outside > 0.85 and 0.25 < inside < 0.75
