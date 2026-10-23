"""Metrics, evaluation machinery and the neural Cox loss."""
import numpy as np
import pandas as pd
import pytest
from sksurv.metrics import concordance_index_censored

from exprkit.evaluation import (assign_risk_groups, batch_separability, confounding_table, cross_validate,
                                make_splits)
from exprkit.metrics import (corrected_resampled_t, cox_neg_log_partial_likelihood, harrell_c,
                             paired_bootstrap)
from exprkit.models import build_pipeline
from exprkit.simulate import simulate_expression, simulate_survival


def _clf(X):
    return lambda: build_pipeline("classification", {"type": "logistic"}, list(X.columns), [], [],
                                  {"top_k_variance": 50})


def test_repeated_cv_tests_every_minority_sample_each_repeat():
    X, y = simulate_expression(50, 10, 100, n_signal=10, effect=1.5, seed=0)
    y = np.where(y == 1, "cancer", "normal")
    splits = make_splits("classification", y, None, {"n_splits": 5, "n_repeats": 3}, 0)
    cv = cross_validate("classification", _clf(X), X, y, None, splits, ["cancer", "normal"], "cancer")
    for p in cv["preds"].values():
        assert not np.isnan(p).any()                          # every sample predicted once per repeat
    assert cv["folds"]["score"].mean() > 0.9


def test_no_signal_gives_chance_auc():
    X, y = simulate_expression(50, 10, 100, n_signal=0, seed=1)
    y = np.where(y == 1, "cancer", "normal")
    splits = make_splits("classification", y, None, {"n_splits": 5, "n_repeats": 4}, 0)
    cv = cross_validate("classification", _clf(X), X, y, None, splits, ["cancer", "normal"], "cancer")
    assert 0.25 < cv["folds"]["score"].mean() < 0.75


def test_grouped_splits_never_share_groups():
    y = np.tile(["a", "b"], 60)
    groups = np.repeat(np.arange(40), 3)
    for _, _, tr, te in make_splits("classification", y, groups, {"n_splits": 4, "n_repeats": 2}, 0):
        assert not set(groups[tr]) & set(groups[te])


def test_batch_check_and_confounding():
    A, _ = simulate_expression(60, 0, 80, n_signal=0, seed=2)
    B, _ = simulate_expression(60, 0, 80, n_signal=0, seed=3)
    B.index = [f"B{i}" for i in range(60)]
    X = pd.concat([A, B + 1.0])
    batch = pd.Series(["A"] * 60 + ["B"] * 60, index=X.index)
    assert batch_separability(X, batch)["auc_predicting_batch"] > 0.95
    _, warns = confounding_table(pd.Series(["normal"] * 10 + ["cancer"] * 110), pd.Series(["A"] * 70 + ["B"] * 50))
    assert len(warns) == 1 and "normal" in warns[0]


# Survival metrics -----------------------------------------------------------------

@pytest.fixture(scope="module")
def surv():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    t, e, log_h = simulate_survival(X, np.array([0.8, -0.5, 0.0]), seed=1)
    return t, e, log_h


def test_harrell_c_matches_scikit_survival(surv):
    t, e, log_h = surv
    assert harrell_c(t, e, log_h) == pytest.approx(concordance_index_censored(e.astype(bool), t, log_h)[0], abs=1e-12)
    assert harrell_c(t, e, -log_h) == pytest.approx(1 - harrell_c(t, e, log_h))


def test_events_only_c_index_differs():
    """A common shortcut, using only pairs where both patients had an event, gives a different answer."""
    t = np.array([1, 2, 3, 4.0]); e = np.array([1, 0, 1, 0]); r = np.array([4, 1, 3, 5.0])

    def events_only_c(t, e, r):
        conc = tot = 0
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                if e[i] == 1 and e[j] == 1:
                    tot += 1
                    conc += (r[i] > r[j] and t[i] < t[j]) or (r[i] < r[j] and t[i] > t[j])
        return conc / tot
    assert harrell_c(t, e, r) == 0.5 and events_only_c(t, e, r) == 1.0


def test_paired_bootstrap_and_corrected_t(surv):
    t, e, log_h = surv
    noisy = log_h + np.random.default_rng(2).normal(0, 2, len(log_h))
    d = paired_bootstrap(lambda i: harrell_c(t[i], e[i], noisy[i]), lambda i: harrell_c(t[i], e[i], log_h[i]),
                         len(t), 200)
    assert d["difference"] > 0 and d["ci_low"] > 0
    rng = np.random.default_rng(0)
    null = corrected_resampled_t(rng.normal(0, 0.02, 50), 160, 40)
    shifted = corrected_resampled_t(rng.normal(0.05, 0.02, 50), 160, 40)
    assert null["p_value"] > 0.05 and shifted["p_value"] < 0.01
    plain_se = 0.02 / np.sqrt(50)
    assert (shifted["ci_high"] - shifted["ci_low"]) / 3.92 > 2 * plain_se   # wider than the naive interval


def test_risk_groups_follow_cutpoints():
    assert assign_risk_groups(np.array([-1, 0.5, 2.0]), [0.0, 1.0]).tolist() == ["low", "middle", "high"]


# Neural Cox loss ------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_torch_loss_matches_reference_with_ties():
    from exprkit.deepsurv import cox_ph_loss
    rng = np.random.default_rng(3)
    t = rng.integers(1, 20, 60).astype(float)
    e = rng.integers(0, 2, 60)
    r = rng.normal(size=60)
    got = cox_ph_loss(torch.tensor(r), torch.tensor(t), torch.tensor(e)).item()
    assert got == pytest.approx(cox_neg_log_partial_likelihood(t, e, r), rel=1e-9)


def test_loss_prefers_true_direction(surv):
    from exprkit.deepsurv import cox_ph_loss
    t, e, log_h = surv
    f = lambda a: torch.tensor(a, dtype=torch.float64)
    assert cox_ph_loss(f(log_h), f(t), f(e)) < cox_ph_loss(f(-log_h), f(t), f(e))


def test_sign_flip_heuristic_fires_on_good_models(surv):
    """A rule that negates scores and flips back when C < 0.4 fires on good models:
    the negated C of a good model is low."""
    t, e, log_h = surv
    assert harrell_c(t, e, -log_h) < 0.4


def test_deepsurv_estimator_learns():
    from exprkit.deepsurv import DeepSurvEstimator
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5))
    t, e, log_h = simulate_survival(X, np.array([1.0, -0.7, 0, 0, 0]), seed=4)
    y = np.array(list(zip(e.astype(bool), t)), dtype=[("event", bool), ("time", float)])
    est = DeepSurvEstimator(hidden=(16,), dropout=0.1, lr=1e-2, max_epochs=300, seed=0).fit(X[:500], y[:500])
    c = est.score(X[500:], y[500:])
    assert c > 0.65 and c > harrell_c(t[500:], e[500:], log_h[500:]) - 0.08


def test_exact_ci_for_small_class():
    from exprkit.metrics import exact_proportion_ci
    lo, hi = exact_proportion_ci(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10), abs=1e-6)   # about 0.69
