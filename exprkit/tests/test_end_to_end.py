"""The full command line workflow on simulated files laid out like GEO downloads:
prepare -> train (classification, survival, regression) -> predict -> evaluate,
plus combine and batch-check."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from exprkit.cli import main
from exprkit.io import load_dataset

from conftest import prepare_cfg_a, prepare_cfg_b


def _run(tmp_path, name, cfg, cmd):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    main([cmd, str(p)])


@pytest.fixture(scope="module")
def prepared(fake_dir, tmp_path_factory):
    raw, _ = fake_dir
    t = tmp_path_factory.mktemp("work")
    _run(t, "prep_a", prepare_cfg_a(raw, t / "A"), "prepare")
    _run(t, "prep_b", prepare_cfg_b(raw, t / "B"), "prepare")
    _run(t, "prep_c", {"output": str(t / "C"), "sources": [{
        "name": "SIM_C",
        "expression": {"path": str(raw / "SIM_C_expression.csv"), "format": "table"},
        "metadata": {"sample_sheet": {"path": str(raw / "SIM_C_samples.csv"), "id_column": "sample_id"}},
        "columns": {"label": {"type": "category", "from": "tissue",
                              "rules": {"normal": "(?i)normal", "cancer": "(?i)tumou?r"}}}}]}, "prepare")
    return t


def test_prepared_datasets(prepared):
    XA, SA, rA = load_dataset(prepared / "A")
    assert XA.shape == (70, 400) and SA["label"].value_counts().to_dict() == {"cancer": 60, "normal": 10}
    assert rA["sources"][0]["mapping"]["n_unmatched"] == 2
    XB, SB, rB = load_dataset(prepared / "B")
    assert XB.shape[0] == 620 and SB.index[0].startswith("GSMB")       # titles linked to GSM IDs
    assert SB["event"].isna().sum() >= 15 and SB["patient"].nunique() == 600
    XC, SC, _ = load_dataset(prepared / "C")
    assert XC.shape == (52, 395)


def test_classification_train_predict_evaluate(prepared):
    t = prepared
    _run(t, "train_clf", {
        "data": str(t / "A"), "output": str(t / "clf"), "task": "classification",
        "target": {"column": "label", "positive_class": "cancer"},
        "genes": {"filter": {"top_k_variance": 200}},
        "model": {"type": "logistic", "grid": {"C": [0.01, 0.1]}},
        "evaluation": {"type": "repeated_cv", "n_splits": 5, "n_repeats": 3, "bootstrap": 200, "permutations": 10},
        "checks": {"batch_column": "dataset"}}, "train")
    s = json.loads((t / "clf" / "summary.json").read_text())
    assert s["class_counts"] == {"cancer": 60, "normal": 10}
    assert s["evaluation"]["fold_score_corrected_interval"]["mean"] > 0.9
    assert s["permutation_test"]["p_value"] <= 1 / 11 + 1e-9
    for f in ("model.joblib", "folds.csv", "oof_predictions.csv", "coefficients.csv", "roc.png",
              "feature_stability.csv"):
        assert (t / "clf" / f).exists()
    # signal genes (G00000 to G00024, 25 of 400) should be strongly enriched among the
    # most stable features: chance would put about 1.25 of them in the top 20
    stab = pd.read_csv(t / "clf" / "feature_stability.csv", index_col=0)
    top = [g.split("__")[1] for g in stab.index[:20]]
    assert sum(int(g[1:]) < 25 for g in top) >= 10
    ci = s["evaluation"]["fold_score_corrected_interval"]
    assert 0 <= ci["ci_low"] <= ci["ci_high"] <= 1

    main(["predict", "--model", str(t / "clf" / "model.joblib"), "--data", str(t / "C"),
          "--out", str(t / "pred_c.csv")])
    pred = pd.read_csv(t / "pred_c.csv", index_col=0)
    assert {"prob_cancer", "prob_normal", "predicted_class"} <= set(pred.columns) and len(pred) == 52
    rep = json.loads((t / "pred_c_input_report.json").read_text())
    assert rep["scale_check"]["warnings"] == []

    main(["evaluate", "--model", str(t / "clf" / "model.joblib"), "--data", str(t / "C"),
          "--out", str(t / "ext"), "--bootstrap", "200"])
    ext = json.loads((t / "ext" / "external_evaluation.json").read_text())
    assert ext["bootstrap_auc"]["auc"] > 0.9


def test_predict_from_matrix_with_scale_warning(prepared, fake_dir, capsys):
    raw, _ = fake_dir
    t = prepared
    expr = pd.read_csv(raw / "SIM_C_expression.csv", index_col=0)
    (2 ** expr).to_csv(t / "unlogged.csv")             # same data, not logged
    main(["predict", "--model", str(t / "clf" / "model.joblib"), "--matrix", str(t / "unlogged.csv"),
          "--out", str(t / "pred_unlogged.csv")])
    assert "WARNING" in capsys.readouterr().out
    main(["predict", "--model", str(t / "clf" / "model.joblib"), "--matrix", str(t / "unlogged.csv"),
          "--transform", "log2", "--out", str(t / "pred_logged.csv")])
    a = pd.read_csv(t / "pred_logged.csv", index_col=0)
    b = pd.read_csv(t / "pred_c.csv", index_col=0)
    assert np.allclose(a["prob_cancer"], b["prob_cancer"], atol=1e-4)


def test_survival_with_covariate_baseline(prepared):
    t = prepared
    _run(t, "train_surv", {
        "data": str(t / "B"), "output": str(t / "surv"), "task": "survival",
        "target": {"time": "time", "event": "event"},
        "group_column": "patient", "one_per_group": True,
        "genes": {"filter": {"top_k_variance": 200}},
        "covariates": {"numeric": ["age", "tumour_size"]},
        "model": {"type": "coxnet", "grid": {"alpha": [0.05, 0.2]}},
        "evaluation": {"type": "holdout", "test_fraction": 0.25, "bootstrap": 200, "permutations": 0}}, "train")
    s = json.loads((t / "surv" / "summary.json").read_text())
    assert s["rows"]["missing_or_invalid_event"] >= 15   # 15 plus replicates of those samples
    assert s["rows"]["dropped_extra_profiles_per_group"] > 0
    c = s["evaluation"]["bootstrap_c_test"]
    assert c["c"] > 0.55 and c["ci_low"] > 0.5            # weak simulated signal, but above chance
    d = s["covariates_only_baseline"]["difference_model_minus_baseline"]
    assert d["ci_low"] <= d["difference"] <= d["ci_high"] and d["method"] == "paired bootstrap"
    main(["predict", "--model", str(t / "surv" / "model.joblib"), "--data", str(t / "B"),
          "--out", str(t / "pred_surv.csv")])
    p = pd.read_csv(t / "pred_surv.csv", index_col=0)
    assert set(p["risk_group"]) == {"low", "middle", "high"}


def test_regression_and_cv_survival(prepared):
    t = prepared
    _run(t, "train_reg", {
        "data": str(t / "B"), "output": str(t / "reg"), "task": "regression",
        "target": {"column": "tumour_size"}, "genes": {"filter": {"top_k_variance": 100}},
        "model": {"type": "ridge", "grid": {"alpha": [1.0, 100.0]}},
        "evaluation": {"type": "repeated_cv", "n_splits": 3, "n_repeats": 1, "bootstrap": 100}}, "train")
    s = json.loads((t / "reg" / "summary.json").read_text())
    assert abs(s["evaluation"]["fold_score_corrected_interval"]["mean"]) < 0.2   # size is not driven by genes here


def test_combine_and_batch_check(prepared, capsys):
    t = prepared
    main(["combine", "--data", str(t / "A"), "--data", str(t / "B"), "--out", str(t / "AB")])
    X, S, _ = load_dataset(t / "AB")
    assert set(S["dataset"]) == {"SIM_A", "SIM_B"}
    main(["batch-check", "--data", str(t / "AB"), "--where", "label=cancer", "--out", str(t / "batch.json")])
    assert json.loads((t / "batch.json").read_text())["auc_predicting_batch"] > 0.95
