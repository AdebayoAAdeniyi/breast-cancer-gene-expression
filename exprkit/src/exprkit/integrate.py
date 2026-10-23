"""Combine prepared datasets on exact gene symbols.

No fuzzy matching: ``difflib.SequenceMatcher`` scores ESR1 against ESR2 at
0.75 and HOXA1 against HOXA2 at 0.80, so approximate matching can silently
replace one gene by another. Missing genes stay missing (NaN); filling, if
any, happens inside a model pipeline on training data only.
"""
from __future__ import annotations

import pandas as pd

from .mapping import normalise_symbol


def _clean_columns(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t.columns = normalise_symbol(t.columns)
    if t.columns.duplicated().any():
        t = t.T.groupby(level=0).mean().T
    return t


def combine_expression(tables: dict[str, pd.DataFrame], how: str = "inner") -> tuple[pd.DataFrame, pd.Series, dict]:
    """Stack samples x genes tables. Returns expression, a ``dataset`` Series and a report."""
    if how not in ("inner", "outer"):
        raise ValueError("how must be 'inner' or 'outer'")
    cleaned = {k: _clean_columns(v) for k, v in tables.items()}
    sets = [set(t.columns) for t in cleaned.values()]
    genes = sorted(set.intersection(*sets)) if how == "inner" else sorted(set.union(*sets))
    frames = [t.reindex(columns=genes) for t in cleaned.values()]
    expr = pd.concat(frames, axis=0)
    if expr.index.duplicated().any():
        raise ValueError(f"sample IDs repeat across datasets, e.g. {expr.index[expr.index.duplicated()][:5].tolist()}")
    dataset = pd.concat([pd.Series(k, index=t.index) for k, t in cleaned.items()])
    report = {"how": how, "genes_per_dataset": {k: int(t.shape[1]) for k, t in cleaned.items()},
              "genes_out": len(genes), "samples_per_dataset": {k: int(t.shape[0]) for k, t in cleaned.items()}}
    return expr, dataset.rename("dataset"), report


def align_to_genes(expr: pd.DataFrame, genes, max_missing_fraction: float = 0.2) -> tuple[pd.DataFrame, dict]:
    """Reindex samples x genes to a fixed gene list by exact symbol (NaN if absent)."""
    e = _clean_columns(expr)
    target = normalise_symbol(genes)
    present = target.intersection(e.columns)
    missing = sorted(set(target) - set(present))
    frac_missing = len(missing) / len(target) if len(target) else 0.0
    rep = {"n_genes_required": int(len(target)), "n_found": int(len(present)),
           "fraction_missing": round(frac_missing, 4), "missing_examples": missing[:25]}
    if frac_missing > max_missing_fraction:
        raise ValueError(f"{frac_missing:.1%} of required genes are missing (limit {max_missing_fraction:.0%}). "
                         f"First missing: {missing[:10]}")
    return e.reindex(columns=target), rep
