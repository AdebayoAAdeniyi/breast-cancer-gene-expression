"""Tables with embedded labels, duplicate profiles, class balance without copies,
per sample rank normalisation, leave one study out, and select-genes end to end."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from exprkit.balance import UndersampleEnsemble
from exprkit.cli import main
from exprkit.io import load_dataset
from exprkit.models import build_pipeline
from exprkit.prepare import find_duplicate_profiles
from exprkit.simulate import simulate_expression
from exprkit.transform import per_sample_rank


def test_rank_normalisation_is_per_sample_and_scale_free():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(5, 2, (4, 50)))
    a = per_sample_rank(X)
    b = per_sample_rank(np.exp(X) * 3 + 10)        # any increasing transform of a sample gives the same result
    assert np.allclose(a, b)
    c = per_sample_rank(X.iloc[[0]])              # one sample alone gets the same values
    assert np.allclose(a.iloc[[0]], c)


def test_duplicate_profiles_found():
    X = pd.DataFrame([[1.0, 2.0], [1.0, 2.0 + 1e-9], [3.0, 4.0]], index=["a", "b", "c"])
    d = find_duplicate_profiles(X)
    assert d.tolist() == ["a", "a", "c"]


def test_undersample_ensemble_balances_without_copies():
    X, y = simulate_expression(90, 10, 60, n_signal=5, effect=1.5, seed=1)
    yl = np.where(y == 1, "tumour", "normal")
    est = build_pipeline("classification", {"type": "logistic"}, list(X.columns), [], [], {}, 0,
                         balance={"method": "undersample", "n_bags": 5})
    est.fit(X, yl)
    assert isinstance(est, UndersampleEnsemble) and est.bag_sizes_ == {"normal": 10, "tumour": 10}
    p = est.predict_proba(X)
    assert p.shape == (100, 2) and np.allclose(p.sum(axis=1), 1)


def _write(tmp, name, obj):
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(obj, sort_keys=False))
    return str(p)


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    """Two 'studies' on different scales, each with both classes, written as
    samples-by-genes tables with a 0/1 Label column and no sample IDs; study 1
    also contains copied normals, as produced by naive oversampling."""
    t = tmp_path_factory.mktemp("tables")
    X1, y1 = simulate_expression(80, 12, 120, n_signal=8, effect=1.3, seed=2, gene_seed=9)
    X2, y2 = simulate_expression(60, 20, 120, n_signal=8, effect=1.3, seed=3, gene_seed=9)
    X2 = X2 * 2.5 + 4                                      # another platform scale
    d1 = X1.assign(Label=y1)
    copies = d1[d1.Label == 0].sample(40, replace=True, random_state=0)
    pd.concat([d1, copies]).to_csv(t / "study1.csv", index=False)
    X2.assign(Label=y2).to_csv(t / "study2.csv", index=False)
    src = lambda n: {"name": n, "expression": {"path": str(t / f"{n}.csv"), "format": "table",
                                               "orientation": "samples_by_genes", "index_column": None,
                                               "metadata_columns": ["Label"]},
                     "columns": {"label": {"type": "category", "from": "Label", "map": {0: "normal", 1: "tumour"}}}}
    main(["prepare", _write(t, "prep", {"output": str(t / "both"), "duplicates": "drop", "normalise": "per_sample_rank",
                                        "sources": [src("study1"), src("study2")]})])
    return t


def test_prepare_tables_drops_copies(tables):
    X, S, r = load_dataset(tables / "both")
    assert r["duplicates"]["n_duplicate_rows"] == 40
    assert pd.crosstab(S.label, S.dataset).loc["normal"].tolist() == [12, 20]


def test_leave_one_study_out_with_selection_and_balance(tables):
    t = tables
    main(["train", _write(t, "logo", {
        "data": str(t / "both"), "output": str(t / "logo"), "task": "classification",
        "target": {"column": "label", "positive_class": "tumour"},
        "genes": {"filter": {}, "select": {"n_genes": 20, "max_correlation": 0.8}},
        "balance": {"method": "undersample", "n_bags": 5},
        "model": {"type": "logistic"},
        "evaluation": {"type": "leave_one_group_out", "column": "dataset", "bootstrap": 100}})])
    s = json.loads((t / "logo" / "summary.json").read_text())
    per = s["evaluation"]["per_held_out_group"]
    assert set(per) == {"study1", "study2"} and all(v["score"] > 0.85 for v in per.values())
    assert s["balance"]["rows_per_class_in_each_bag"] == {"normal": 32, "tumour": 32}
    assert s["final_model"]["genes_after_filter"] >= 20      # union of selected genes across bags


def test_select_genes_command(tables, tmp_path):
    t = tables
    main(["select-genes", _write(tmp_path, "sel", {
        "data": str(t / "both"), "split_by": "dataset", "output": str(tmp_path / "sel"),
        "task": "classification", "target": {"column": "label", "positive_class": "tumour"},
        "n_genes": 8, "max_correlation": 0.9, "stability": {"n_boot": 10, "top": 50}})])
    genes = (tmp_path / "sel" / "genes.txt").read_text().split()
    assert len(genes) == 8 and sum(int(g[1:]) < 8 for g in genes) >= 7    # the 8 planted signal genes
    tab = pd.read_csv(tmp_path / "sel" / "gene_selection.csv", index_col=0)
    assert {"z_study1", "z_study2", "combined_z", "sign_agreement", "stability", "selected"} <= set(tab.columns)


def test_predict_applies_training_rank_normalisation(tables):
    t = tables
    src = {"name": "study2", "expression": {"path": str(t / "study2.csv"), "format": "table",
                                            "orientation": "samples_by_genes", "index_column": None,
                                            "metadata_columns": ["Label"]},
           "columns": {"label": {"type": "category", "from": "Label", "map": {0: "normal", 1: "tumour"}}}}
    main(["prepare", _write(t, "prep2", {"output": str(t / "s2raw"), "sources": [src]})])
    main(["evaluate", "--model", str(t / "logo" / "model.joblib"), "--data", str(t / "s2raw"),
          "--out", str(t / "ext2"), "--bootstrap", "50"])
    ext = json.loads((t / "ext2" / "external_evaluation.json").read_text())
    assert "normalise" in ext["input_report"] and ext["bootstrap_auc"]["auc"] > 0.85
    with pytest.raises(SystemExit):                 # already normalised data are refused
        main(["predict", "--model", str(t / "logo" / "model.joblib"), "--data", str(t / "both"),
              "--out", str(t / "p.csv")])
