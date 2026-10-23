"""Reading input files and reading or writing prepared datasets.

Input formats
-------------
* GEO series matrix (``GSExxxx_series_matrix.txt[.gz]``). ``!Sample_`` lines
  hold per sample metadata; the block between ``!series_matrix_table_begin``
  and ``!series_matrix_table_end`` holds the expression table.
* Plain expression tables (CSV or TSV, optionally gzipped), either genes by
  samples or samples by genes.
* Annotation tables that map feature IDs (probes, transcripts, Ensembl IDs)
  to gene symbols, including GEO platform "full table" downloads.

Prepared dataset (the output of ``exprkit prepare``)
----------------------------------------------------
A folder with
* ``expression.parquet``: samples in rows, gene symbols in columns, float32.
* ``samples.csv``: one row per sample (same order), derived columns such as
  labels, survival time, event, covariates, patient group, ``dataset``.
* ``report.json``: everything that was dropped, mapped or checked.
"""
from __future__ import annotations

import gzip
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# Generic helpers -------------------------------------------------------------

def open_text(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def guess_sep(path: str | Path, sep: str | None = None) -> str:
    if sep:
        return sep
    name = str(path).lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return "," if name.endswith(".csv") else "\t"


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def to_number(value) -> float:
    """First number in a text value, decimals kept; NaN if there is none."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    m = _NUM.search(str(value))
    return float(m.group()) if m else float("nan")


# GEO series matrix -------------------------------------------------------------

def is_series_matrix(path: str | Path) -> bool:
    return "series_matrix" in Path(path).name.lower()


def read_series_matrix(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a GEO series matrix.

    Returns ``(expression, metadata)``. ``expression`` has features in rows and
    GSM accessions in columns (empty when the series stores its data in
    supplementary files). ``metadata`` has one row per GSM. Characteristic
    lines ``key: value`` become columns named ``char:<key>`` (lower case).
    """
    meta_raw: dict[str, list] = {}
    seen: dict[str, int] = defaultdict(int)
    table_lines: list[str] = []
    in_table = False
    with open_text(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table_lines.append(line)
                continue
            if not line.startswith("!Sample_"):
                continue
            key = line.split("\t", 1)[0][len("!Sample_"):]
            values = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            if key == "characteristics_ch1":
                per_key: dict[str, list] = defaultdict(lambda: [None] * len(values))
                for i, v in enumerate(values):
                    if ":" in v:
                        k, val = v.split(":", 1)
                        per_key["char:" + k.strip().lower()][i] = val.strip()
                for k, vals in per_key.items():
                    if k in meta_raw:
                        meta_raw[k] = [a if a not in (None, "") else b for a, b in zip(meta_raw[k], vals)]
                    else:
                        meta_raw[k] = vals
                continue
            seen[key] += 1
            meta_raw[key if seen[key] == 1 else f"{key}_{seen[key]}"] = values

    if "geo_accession" not in meta_raw:
        raise ValueError(f"{path}: no !Sample_geo_accession line; is this a GEO series matrix?")
    metadata = pd.DataFrame(meta_raw)
    metadata.index = metadata.pop("geo_accession").astype(str)
    metadata.index.name = "sample"
    if table_lines:
        expression = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
        expression.index = expression.index.astype(str)
        expression.columns = [str(c).strip('"') for c in expression.columns]
        expression = expression.apply(pd.to_numeric, errors="coerce")
    else:
        expression = pd.DataFrame()
    return expression, metadata


# Tables ------------------------------------------------------------------------------

def read_matrix(path: str | Path, orientation: str = "genes_by_samples", sep: str | None = None,
                index_column: int | str | None = 0,
                metadata_columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Read an expression table and return samples x features (float32).

    * ``index_column``: column with feature IDs (genes by samples) or sample IDs
      (samples by genes). ``None`` for a samples by genes file without sample
      IDs: rows are then named ``<file stem>_<row number>``.
    * ``metadata_columns`` (samples by genes only): columns that hold sample
      information rather than expression, such as a label, age or survival
      time. They are returned separately in the report under ``"metadata"``.
    * Other non numeric columns (for example a gene description column) are
      dropped and reported.
    """
    df = pd.read_csv(path, sep=guess_sep(path, sep), index_col=index_column, low_memory=False)
    if index_column is None:
        stem = Path(path).name.split(".")[0]
        df.index = [f"{stem}_{i + 1}" for i in range(len(df))]
    df.index = pd.Index(df.index.astype(str)).str.strip().str.strip('"')
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    meta = None
    if metadata_columns:
        if orientation != "samples_by_genes":
            raise ValueError("metadata_columns needs orientation: samples_by_genes")
        missing = [c for c in metadata_columns if c not in df.columns]
        if missing:
            raise KeyError(f"{path}: metadata columns {missing} not found. First columns: {list(df.columns[:10])}")
        meta = df[metadata_columns].copy()
        meta.index.name = "sample"
        df = df.drop(columns=metadata_columns)
    if df.index.duplicated().any():
        raise ValueError(f"{path}: duplicated IDs, e.g. {df.index[df.index.duplicated()][:5].tolist()}")

    def _numeric(d):
        if all(pd.api.types.is_numeric_dtype(t) for t in d.dtypes):
            return d                                  # fast path: already numeric
        return d.apply(pd.to_numeric, errors="coerce")

    if orientation not in ("genes_by_samples", "samples_by_genes"):
        raise ValueError("orientation must be 'genes_by_samples' or 'samples_by_genes'")
    numeric = _numeric(df)
    dropped = [c for c in numeric.columns if numeric[c].isna().all() and not df[c].isna().all()]
    numeric = numeric.drop(columns=dropped)
    X = numeric.T if orientation == "genes_by_samples" else numeric
    X = X.astype("float32")
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    return X, {"non_numeric_columns_dropped": dropped, "metadata": meta}


def read_annotation(path: str | Path, id_column: str = "ID", symbol_column: str = "GENE_SYMBOL",
                    sep: str | None = None) -> pd.DataFrame:
    """Read a feature to gene symbol table. Returns columns ``probe_id`` and ``symbol``.

    Lines starting with ``#``, ``!`` or ``^`` are skipped before parsing, so a
    ``#`` inside a field (common in sequence columns) is kept intact.
    """
    with open_text(path) as fh:
        body = [ln for ln in fh if not ln.startswith(("#", "!", "^"))]
    table = pd.read_csv(io.StringIO("".join(body)), sep=guess_sep(path, sep), dtype=str, keep_default_na=False)
    missing = {id_column, symbol_column} - set(table.columns)
    if missing:
        raise KeyError(f"{path}: columns {sorted(missing)} not found. Available: {list(table.columns)[:30]}")
    out = table[[id_column, symbol_column]].rename(columns={id_column: "probe_id", symbol_column: "symbol"})
    out["probe_id"] = out["probe_id"].str.strip()
    out["symbol"] = out["symbol"].str.strip()
    return out


def read_sample_sheet(path: str | Path, id_column: str, sep: str | None = None) -> pd.DataFrame:
    sheet = pd.read_csv(path, sep=guess_sep(path, sep), dtype=str, keep_default_na=False, na_values=[""])
    if id_column not in sheet.columns:
        raise KeyError(f"{path}: id column '{id_column}' not found. Available: {list(sheet.columns)}")
    sheet.index = sheet.pop(id_column).astype(str).str.strip()
    sheet.index.name = "sample"
    if sheet.index.duplicated().any():
        raise ValueError(f"{path}: sample IDs repeat, e.g. {sheet.index[sheet.index.duplicated()][:5].tolist()}")
    return sheet


# Prepared datasets ------------------------------------------------------------------

def save_dataset(directory: str | Path, expression: pd.DataFrame, samples: pd.DataFrame, report: dict) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if not expression.index.equals(samples.index):
        raise ValueError("expression and samples must have the same index in the same order")
    expr = expression.astype("float32").copy()
    expr.columns = expr.columns.astype(str)
    expr.index = expr.index.astype(str)
    expr.index.name = "sample"
    expr.to_parquet(d / "expression.parquet")
    s = samples.copy()
    s.index.name = "sample"
    s.to_csv(d / "samples.csv")
    write_json(report, d / "report.json")
    return d


def load_dataset(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = Path(directory)
    if not (d / "expression.parquet").exists():
        raise FileNotFoundError(f"{d} is not a prepared dataset (no expression.parquet). Run 'exprkit prepare' first.")
    expr = pd.read_parquet(d / "expression.parquet")
    samples = pd.read_csv(d / "samples.csv", index_col=0, dtype={"sample": str})
    samples.index = samples.index.astype(str)
    expr.index = expr.index.astype(str)
    if not expr.index.equals(samples.index):
        raise ValueError(f"{d}: expression and samples.csv do not list the same samples in the same order")
    report = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else {}
    return expr, samples, report
