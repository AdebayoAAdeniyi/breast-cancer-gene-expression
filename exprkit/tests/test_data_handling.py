"""Reading, mapping, sample columns, integration and filtering.
Each test pins down one behaviour that is easy to get wrong in expression analyses."""
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import pytest

from exprkit.filtering import ExpressionFilter, duplicate_columns, mode_fraction
from exprkit.integrate import align_to_genes, combine_expression
from exprkit.io import read_annotation, read_matrix, read_series_matrix, to_number
from exprkit.mapping import map_to_symbols
from exprkit.samples import SpecError, category_from_rules, derive_columns, parse_event, patient_groups
from exprkit.transform import apply_transform, scale_warnings


def test_series_matrix_reader(fake_dir):
    d, info = fake_dir
    expr, meta = read_series_matrix(d / "SIM_A_series_matrix.txt")
    assert meta.shape[0] == 70
    assert list(expr.columns) == list(meta.index)
    _, meta_b = read_series_matrix(d / "SIM_B-GPL1_series_matrix.txt")
    assert {"char:overall survival days", "char:overall survival event", "char:tumor size"} <= set(meta_b.columns)


def test_annotation_reader_keeps_hash_inside_fields(fake_dir):
    d, _ = fake_dir
    ann = read_annotation(d / "SIM_A_platform.txt")
    assert len(ann) > 400
    raw = pd.read_csv(d / "SIM_A_platform.txt", sep="\t", comment="#")
    assert raw["SEQUENCE"].iloc[0] == "ACGT"          # pandas comment="#" cuts "ACGT#ACGT"


def test_read_matrix_orientation(fake_dir):
    d, _ = fake_dir
    X, _ = read_matrix(d / "SIM_B_gene_expression.csv", "genes_by_samples")
    assert X.shape == (620, 400) and X.index[0] == "F0"


def test_mapping_reports_unmatched_and_ambiguous(fake_dir):
    d, _ = fake_dir
    expr, _ = read_series_matrix(d / "SIM_A_series_matrix.txt")
    genes, rep = map_to_symbols(expr, read_annotation(d / "SIM_A_platform.txt"))
    assert rep.n_unmatched == 2 and set(rep.unmatched_ids) == {"P_CTRL_1", "P_CTRL_2"}
    assert rep.n_ambiguous == 1 and genes.shape[0] == 400


def test_mapping_averages_duplicate_probes():
    expr = pd.DataFrame({"s1": [1.0, 3.0, 10.0]}, index=["p1", "p2", "p3"])
    ann = pd.DataFrame({"probe_id": ["p1", "p2", "p3"], "symbol": ["ESR1", "esr1 ", "ERBB2"]})
    genes, rep = map_to_symbols(expr, ann)
    assert genes.loc["ESR1", "s1"] == 2.0 and rep.n_genes_with_multiple_ids == 1


def test_labels_from_metadata_not_position(fake_dir):
    d, _ = fake_dir
    _, meta = read_series_matrix(d / "SIM_A_series_matrix.txt")
    lab = category_from_rules(meta["source_name_ch1"], {"normal": "(?i)normal", "cancer": "(?i)tumou?r"},
                              {"normal": 10, "cancer": 60})
    positional = np.array(["cancer"] * 60 + ["normal"] * 10)   # labelling by position would assume the last 10 are normal
    assert (lab.to_numpy() != positional).sum() > 0


def test_labels_fail_loudly():
    text = pd.Series(["normal breast", "tumor", "normal tissue adjacent to tumor", "?"])
    with pytest.raises(SpecError):
        category_from_rules(text, {"normal": "normal", "cancer": "tumor"})
    with pytest.raises(SpecError):
        category_from_rules(text.iloc[:2], {"normal": "normal", "cancer": "tumor"}, {"normal": 10})


def test_derive_columns_types():
    meta = pd.DataFrame({"t": ["100", "22.5 days", "NA"], "e": ["1", "0", "unknown"], "title": ["F1", "F2", "F1repl"]},
                        index=["a", "b", "c"])
    out, rep = derive_columns(meta, {"time": {"type": "number", "from": "t"}, "event": {"type": "event", "from": "e"},
                                     "patient": {"type": "group", "from": "title", "strip_regex": "repl$",
                                                 "expected_groups_with_replicates": 1},
                                     "label": {"value": "cancer"}})
    assert out["time"].tolist()[:2] == [100.0, 22.5] and np.isnan(out["time"].iloc[2])
    assert out["event"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["event"].iloc[2])
    assert out["patient"].tolist() == ["F1", "F2", "F1"]
    with pytest.raises(SpecError):
        derive_columns(meta, {"patient": {"type": "group", "from": "title", "expected_groups_with_replicates": 1}})
    with pytest.raises(SpecError):
        derive_columns(meta, {"x": {"type": "number", "from": "not_a_field"}})


def test_fuzzy_matching_would_swap_real_genes():
    """Why exact matching: a typical fuzzy cut off of 0.7 accepts different genes."""
    for a, b in [("ESR1", "ESR2"), ("HOXA1", "HOXA2"), ("CCL3", "CCL4"), ("MAGEA1", "MAGEA2")]:
        assert SequenceMatcher(None, a, b).ratio() >= 0.7


def test_align_exact_and_nan():
    X = pd.DataFrame([[1.0, 2.0]], columns=["ESR2", "ERBB2"])
    out, rep = align_to_genes(X, ["ESR1", "ERBB2"], max_missing_fraction=0.5)
    assert np.isnan(out.loc[0, "ESR1"]) and rep["missing_examples"] == ["ESR1"]
    with pytest.raises(ValueError):
        align_to_genes(X, ["ESR1", "ERBB2"], max_missing_fraction=0.1)


def test_combine_inner_outer():
    A = pd.DataFrame([[1, 2]], columns=["g1", "g2"], index=["a1"])
    B = pd.DataFrame([[3, 4, 5]], columns=["G1", "g2", "g3"], index=["b1"])
    X, ds, _ = combine_expression({"A": A, "B": B}, "inner")
    assert list(X.columns) == ["G1", "G2"] and ds["b1"] == "B"
    X, _, _ = combine_expression({"A": A, "B": B}, "outer")
    assert np.isnan(X.loc["a1", "G3"])


def test_mode_fraction_and_duplicates():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, (50, 20)).astype(float)
    X[:, 0] = 1.0
    ref = [pd.Series(X[:, j]).value_counts(normalize=True).iloc[0] for j in range(20)]
    assert np.allclose(mode_fraction(X), ref)
    X[:, 5] = X[:, 3]
    assert duplicate_columns(X)[5] and not duplicate_columns(X)[3]


def test_filter_fits_on_training_rows_only_and_ignores_nan_variance():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(5, 1, (40, 6)), columns=list("abcdef"))
    X["const"], X["dup"], X["empty"] = 1.0, X["a"], np.nan
    f = ExpressionFilter(top_k_variance=4).fit(X.iloc[:30])
    kept = list(f.get_feature_names_out())
    assert "const" not in kept and "dup" not in kept and "empty" not in kept and len(kept) == 4
    assert f.transform(X.iloc[30:]).shape == (10, 4)


def test_transforms_and_scale_warnings():
    counts = pd.DataFrame([[0, 10, 990], [5, 5, 90]], columns=list("abc"), dtype=float)
    cpm = apply_transform(counts, "log2cpm")
    assert cpm.loc[0, "c"] == pytest.approx(np.log2(990 / 1000 * 1e6 + 1), rel=1e-6)
    with pytest.raises(ValueError):
        apply_transform(counts - 100, "log2p1")
    assert scale_warnings(counts * 100, "none")          # big values left untransformed
    assert scale_warnings(np.log2(counts + 1), "log2p1")  # already logged


def test_number_and_event_parsing():
    assert to_number("22.5") == 22.5 and np.isnan(to_number("NA"))
    assert parse_event("1") == 1 and parse_event("Dead") == 1 and parse_event("0") == 0
    assert np.isnan(parse_event("NA"))


def test_patient_groups():
    titles = pd.Series([f"F{i}" for i in range(100)] + [f"F{i}repl" for i in range(10)])
    _, rep = patient_groups(titles, "repl$")
    assert rep["groups_with_more_than_one_sample"] == 10


def test_na_text_is_missing():
    meta = pd.DataFrame({"er": ["1", "0", "NA", "unknown"], "size": ["20", "NA", "15.5", ""]})
    out, rep = derive_columns(meta, {"er": {"type": "category", "from": "er"}, "size": {"type": "number", "from": "size"}})
    assert out["er"].isna().sum() == 2 and set(out["er"].dropna()) == {"1", "0"}
    assert out["size"].isna().sum() == 2
    with pytest.raises(SpecError):                       # rules cannot classify a missing value
        derive_columns(meta, {"er": {"type": "category", "from": "er", "rules": {"pos": "1", "neg": "0"}}})
