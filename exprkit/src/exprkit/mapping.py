"""Map probe or transcript identifiers to gene symbols and collapse duplicates.

Used by exprkit prepare.

Decisions that are made explicit here (and reported, never silent):

* Probes with no symbol in the platform table are dropped and listed.
* Probes annotated to more than one gene (GEO uses ``" /// "`` as separator)
  are ambiguous. They are dropped by default and counted.
* Several probes for one gene are averaged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MULTI_GENE_PATTERN = r"///|;|,"


def normalise_symbol(s: pd.Index | pd.Series) -> pd.Index:
    """Upper case and strip whitespace. Applied to every symbol before joins."""
    return pd.Index(pd.Series(s, dtype=str).str.strip().str.upper())


@dataclass
class MappingReport:
    n_input_ids: int = 0
    n_unmatched: int = 0
    n_ambiguous: int = 0
    n_genes_out: int = 0
    n_genes_with_multiple_ids: int = 0
    unmatched_ids: list[str] = field(default_factory=list)
    ambiguous_ids: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not isinstance(v, list)}


def map_to_symbols(expression: pd.DataFrame, annotation: pd.DataFrame,
                   drop_ambiguous: bool = True) -> tuple[pd.DataFrame, MappingReport]:
    """Map rows of ``expression`` (probe IDs) to gene symbols.

    Parameters
    ----------
    expression : probes in rows, samples in columns.
    annotation : DataFrame with columns ``probe_id`` and ``symbol`` (see
        :func:`exprkit.io.read_annotation`).

    Returns
    -------
    gene_expression : genes in rows (normalised symbols), samples in columns,
        duplicates averaged.
    report : counts and lists of what was dropped.
    """
    rep = MappingReport(n_input_ids=expression.shape[0])
    ann = annotation.copy()
    ann["probe_id"] = ann["probe_id"].astype(str)
    ann = ann[ann["symbol"].astype(str).str.len() > 0]
    id_to_symbol = ann.drop_duplicates("probe_id").set_index("probe_id")["symbol"]

    symbols = pd.Series(expression.index.astype(str), index=expression.index).map(id_to_symbol)
    unmatched = symbols.isna()
    rep.unmatched_ids = sorted(expression.index[unmatched].astype(str))
    rep.n_unmatched = int(unmatched.sum())

    is_multi = symbols.fillna("").str.contains(MULTI_GENE_PATTERN, regex=True)
    if drop_ambiguous:
        rep.ambiguous_ids = sorted(expression.index[is_multi].astype(str))
        rep.n_ambiguous = int(is_multi.sum())
        keep = ~unmatched & ~is_multi
    else:
        keep = ~unmatched

    expr = expression.loc[keep].copy()
    expr.index = normalise_symbol(symbols[keep])
    counts = expr.index.value_counts()
    rep.n_genes_with_multiple_ids = int((counts > 1).sum())
    gene_expr = expr.groupby(level=0).mean()
    rep.n_genes_out = gene_expr.shape[0]
    return gene_expr, rep


def collapse_duplicate_symbols(expression: pd.DataFrame) -> pd.DataFrame:
    """Normalise symbols in the row index and average rows sharing a symbol."""
    expr = expression.copy()
    expr.index = normalise_symbol(expr.index)
    return expr.groupby(level=0).mean()
