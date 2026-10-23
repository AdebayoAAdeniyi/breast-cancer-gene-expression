import pytest

from exprkit.simulate import write_fake_external, write_fake_geo


@pytest.fixture(scope="session")
def fake_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    info = write_fake_geo(d, seed=1)
    info.update({"external": write_fake_external(d)})
    return d, info


def prepare_cfg_a(raw, out):
    return {"output": str(out), "sources": [{
        "name": "SIM_A",
        "expression": {"path": str(raw / "SIM_A_series_matrix.txt")},
        "id_mapping": {"path": str(raw / "SIM_A_platform.txt"), "id_column": "ID", "symbol_column": "GENE_SYMBOL"},
        "columns": {"label": {"type": "category", "from": "source_name_ch1",
                              "rules": {"normal": "(?i)normal", "cancer": "(?i)tumou?r"},
                              "expected": {"normal": 10, "cancer": 60}}}}]}


def prepare_cfg_b(raw, out):
    return {"output": str(out), "sources": [{
        "name": "SIM_B",
        "expression": {"path": str(raw / "SIM_B_gene_expression.csv"), "format": "table",
                       "orientation": "genes_by_samples"},
        "metadata": {"geo_series_matrix": [str(raw / "SIM_B-GPL1_series_matrix.txt"),
                                           str(raw / "SIM_B-GPL2_series_matrix.txt")]},
        "match_expression_on": "title",
        "columns": {"time": {"type": "number", "from": "char:overall survival days"},
                    "event": {"type": "event", "from": "char:overall survival event"},
                    "age": {"type": "number", "from": "char:age at diagnosis"},
                    "tumour_size": {"type": "number", "from": "char:tumor size"},
                    "patient": {"type": "group", "from": "title", "strip_regex": "repl$",
                                "expected_groups_with_replicates": 20},
                    "label": {"value": "cancer"}}}]}
