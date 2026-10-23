"""Shipped templates and example configs must be valid and use known options."""
from importlib import resources
from pathlib import Path

import yaml

from exprkit.models import MODELS

ROOT = Path(__file__).resolve().parents[1]


def test_templates_parse():
    for kind in ("prepare", "train", "select-genes"):
        cfg = yaml.safe_load(resources.files("exprkit.templates").joinpath(f"{kind}.yaml").read_text())
        assert "output" in cfg


def test_example_configs():
    for p in (ROOT / "examples").rglob("*.yaml"):
        cfg = yaml.safe_load(p.read_text())
        if "task" in cfg and "model" not in cfg:          # select-genes config
            assert cfg["task"] in MODELS and cfg["n_genes"] > 0, p
        elif "task" in cfg:
            assert cfg["model"]["type"] in MODELS[cfg["task"]], p
        else:
            assert cfg["sources"] and all("expression" in s or "metadata" in s for s in cfg["sources"]), p
