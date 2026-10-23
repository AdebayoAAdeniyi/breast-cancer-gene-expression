"""``exprkit prepare``: from raw files to a prepared dataset.

For each source listed in the config:

1. Read expression (GEO series matrix or a table) as samples x features.
2. Read metadata (GEO series matrices and/or a sample sheet) and link it to
   the expression samples, either by ID or through a metadata column
   (``match_expression_on``, e.g. ``title`` when the expression file uses
   sample titles instead of GSM accessions). Unlinked samples are reported.
3. Transform values (``none``, ``log2``, ``log2p1``, ``log2cpm``), before
   mapping, so library sizes use all features and probe averaging happens on
   the log scale.
4. Map feature IDs to gene symbols (if ``id_mapping`` is given) or normalise
   the symbols already present; average features of the same gene.
5. Derive sample columns (labels, survival, covariates, groups) with checks.

Several sources are combined on exact gene symbols (``combine: inner`` or
``outer``) and a ``dataset`` column records each sample's source.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .integrate import combine_expression
from .io import (is_series_matrix, read_annotation, read_matrix, read_sample_sheet,
                 read_series_matrix, save_dataset)
from .mapping import collapse_duplicate_symbols, map_to_symbols
from .samples import derive_columns
from .transform import apply_normalisation, apply_transform, scale_summary, scale_warnings


def _path(p, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


def _read_expression(spec: dict, base: Path):
    path = _path(spec["path"], base)
    fmt = spec.get("format") or ("geo_series_matrix" if is_series_matrix(path) else "table")
    if fmt == "geo_series_matrix":
        expr, meta = read_series_matrix(path)
        if expr.empty:
            raise ValueError(f"{path} has no expression table. For this series the values are probably in a "
                             "supplementary file; use format: table for that file and list the series matrix "
                             "under metadata.")
        return expr.T.astype("float32"), meta, {"format": fmt}
    if fmt == "table":
        X, rep = read_matrix(path, spec.get("orientation", "genes_by_samples"), spec.get("sep"),
                             spec.get("index_column", 0), spec.get("metadata_columns"))
        meta = rep.pop("metadata")
        if meta is not None:
            rep["metadata_columns"] = list(meta.columns)
        return X, meta, {"format": fmt, **rep}
    raise ValueError(f"unknown expression format '{fmt}'")


def _read_metadata(spec: dict | None, meta_from_matrix, base: Path) -> pd.DataFrame | None:
    spec = spec or {}
    parts = []
    geo = spec.get("geo_series_matrix")
    if geo:
        geo = [geo] if isinstance(geo, str) else geo
        parts.append(pd.concat([read_series_matrix(_path(p, base))[1] for p in geo]))
        if meta_from_matrix is not None and not meta_from_matrix.columns.empty:
            # sample columns that came inside an expression table join the GEO metadata
            parts[0] = parts[0].join(meta_from_matrix, how="left", rsuffix="_table")
    elif meta_from_matrix is not None:
        parts.append(meta_from_matrix)
    meta = parts[0] if parts else None
    if meta is not None and meta.index.duplicated().any():
        raise ValueError(f"sample IDs repeat across metadata files: {meta.index[meta.index.duplicated()][:5].tolist()}")
    sheet = spec.get("sample_sheet")
    if sheet:
        s = read_sample_sheet(_path(sheet["path"], base), sheet.get("id_column", "sample"), sheet.get("sep"))
        meta = s if meta is None else meta.join(s, how="left", rsuffix="_sheet")
    return meta


def _link(X: pd.DataFrame, meta: pd.DataFrame | None, key: str | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if meta is None:
        return X, pd.DataFrame(index=X.index), {"metadata": "none"}
    if key:
        if key not in meta.columns:
            raise KeyError(f"match_expression_on: '{key}' is not a metadata field. Available: {sorted(meta.columns)}")
        values = meta[key].astype(str).str.strip()
        if values.duplicated().any():
            raise ValueError(f"metadata field '{key}' repeats values, e.g. {values[values.duplicated()].head(5).tolist()}")
        lookup = dict(zip(values.to_numpy(), meta.index))
        mapped = pd.Series([lookup.get(s) for s in X.index], index=X.index, dtype=object)
    else:
        ids = set(meta.index)
        mapped = pd.Series([s if s in ids else None for s in X.index], index=X.index, dtype=object)
    linked = mapped.notna()
    rep = {"expression_samples": int(len(X)), "metadata_samples": int(len(meta)),
           "linked": int(linked.sum()),
           "expression_without_metadata": X.index[~linked.to_numpy()].tolist()[:50],
           "n_expression_without_metadata": int((~linked).sum())}
    if linked.sum() == 0:
        mvals = meta[key].head(5).tolist() if key else meta.index[:5].tolist()
        raise ValueError(f"no expression sample could be linked to metadata. Expression samples look like "
                         f"{X.index[:5].tolist()}, metadata {'field ' + key if key else 'IDs'} look like {mvals}.")
    X = X.loc[linked.to_numpy()].copy()
    X.index = pd.Index(mapped[linked].astype(str).to_numpy(), name="sample")
    if X.index.duplicated().any():
        raise ValueError(f"several expression samples link to the same metadata sample: "
                         f"{X.index[X.index.duplicated()][:5].tolist()}")
    rep["n_metadata_without_expression"] = int(len(meta.index.difference(X.index)))
    return X, meta.loc[X.index], rep


def prepare_source(src: dict, base: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    name = src["name"]
    rep: dict = {"name": name}
    if not src.get("expression"):
        # Clinical only: no expression yet (or none needed), samples come from metadata.
        meta = _read_metadata(src.get("metadata"), None, base)
        if meta is None:
            raise ValueError(f"source '{name}': without expression, metadata is required")
        samples, rep["columns"] = derive_columns(meta, src.get("columns"))
        rep.update({"expression": None, "transform": "none", "warnings": [],
                    "note": "metadata only; no expression values"})
        return pd.DataFrame(index=meta.index.rename("sample"), dtype="float32"), samples, rep
    X, meta_m, rep["expression"] = _read_expression(src["expression"], base)
    rep["expression"]["shape_as_read"] = list(X.shape)
    meta = _read_metadata(src.get("metadata"), meta_m, base)
    X, meta, rep["linking"] = _link(X, meta, src.get("match_expression_on"))

    kind = src.get("transform", "none")
    rep["transform"] = kind
    rep["scale_before_transform"] = scale_summary(X)
    rep["warnings"] = scale_warnings(X, kind)
    X = apply_transform(X, kind)

    if src.get("id_mapping"):
        m = src["id_mapping"]
        ann = read_annotation(_path(m["path"], base), m.get("id_column", "ID"), m.get("symbol_column", "GENE_SYMBOL"),
                              m.get("sep"))
        genes, mrep = map_to_symbols(X.T, ann, drop_ambiguous=m.get("drop_ambiguous", True))
        X = genes.T
        rep["mapping"] = {**mrep.summary(), "unmatched_examples": mrep.unmatched_ids[:25],
                          "ambiguous_examples": mrep.ambiguous_ids[:25]}
    else:
        n0 = X.shape[1]
        X = collapse_duplicate_symbols(X.T).T
        rep["mapping"] = {"id_mapping": None, "features_in": n0, "genes_out": int(X.shape[1])}
    all_nan = X.columns[X.isna().all()]
    X = X.drop(columns=all_nan).astype("float32")
    rep["genes_all_missing_dropped"] = int(len(all_nan))

    samples, rep["columns"] = derive_columns(meta, src.get("columns"))
    rep["scale_after"] = scale_summary(X)
    return X, samples, rep


def find_duplicate_profiles(X: pd.DataFrame, decimals: int = 6) -> pd.Series:
    """ID of the first sample with an identical expression profile, for every sample.

    Profiles are compared after rounding to ``decimals`` places, so copies that
    differ only by floating point noise are still caught.
    """
    keys = pd.util.hash_pandas_object(X.round(decimals), index=False)
    first = pd.Series(X.index, index=keys.to_numpy()).groupby(level=0).transform("first")
    return pd.Series(first.to_numpy(), index=X.index, name="duplicate_of")


def handle_duplicates(X: pd.DataFrame, S: pd.DataFrame, mode: str = "warn"):
    """Identical expression profiles under different sample IDs.

    They usually mean a sample was copied (for example to balance classes) or
    deposited twice. Copies on both sides of a train/test split make the test
    score optimistic, and copies inside training only repeat information.

    mode: ``warn`` (report only), ``drop`` (keep the first copy), or ``group``
    (keep all, and write a ``duplicate_group`` column to use as group_column).
    """
    if X.shape[1] == 0:
        return X, S, {"n_duplicate_rows": 0}
    dup_of = find_duplicate_profiles(X)
    is_copy = dup_of != X.index
    rep = {"n_duplicate_rows": int(is_copy.sum()), "n_profiles_with_copies": int(dup_of[is_copy].nunique()),
           "mode": mode}
    if is_copy.any():
        sizes = dup_of.value_counts()
        rep["largest_copy_counts"] = sizes.head(10).to_dict()
        print(f"WARNING: {rep['n_duplicate_rows']} samples are exact copies of another sample's expression "
              f"profile ({rep['n_profiles_with_copies']} distinct profiles copied). duplicates: {mode}")
    if mode == "drop":
        X, S = X.loc[~is_copy.to_numpy()], S.loc[~is_copy.to_numpy()]
    elif mode == "group":
        S = S.assign(duplicate_group=dup_of.loc[S.index].to_numpy())
    elif mode != "warn":
        raise ValueError("duplicates must be warn, drop or group")
    return X, S, rep


def prepare(cfg: dict, base: str | Path = ".") -> dict:
    base = Path(base)
    sources = cfg["sources"]
    names = [s["name"] for s in sources]
    if len(set(names)) != len(names):
        raise ValueError("source names must be unique")
    exprs, samples, reports = {}, {}, []
    for src in sources:
        X, S, r = prepare_source(src, base)
        exprs[src["name"]], samples[src["name"]] = X, S
        reports.append(r)
        print(f"[{src['name']}] {X.shape[0]} samples x {X.shape[1]} genes"
              + "".join(f"\n  warning: {w}" for w in r["warnings"]))
    if len(sources) == 1:
        X = exprs[names[0]]
        dataset = pd.Series(names[0], index=X.index, name="dataset")
        comb = None
    else:
        X, dataset, comb = combine_expression(exprs, cfg.get("combine", "inner"))
    S = pd.concat([samples[n] for n in names], axis=0).loc[X.index]
    S["dataset"] = dataset.loc[X.index]
    X, S, report_dup = handle_duplicates(X, S, cfg.get("duplicates", "warn"))
    norm = cfg.get("normalise", "none")
    X = apply_normalisation(X, norm)
    report = {"sources": reports, "combine": comb, "duplicates": report_dup, "normalise": norm, "n_samples": int(X.shape[0]), "n_genes": int(X.shape[1]),
              "transforms": {r["name"]: r["transform"] for r in reports}}
    out = _path(cfg["output"], base)
    save_dataset(out, X, S, report)
    print(f"Prepared dataset: {X.shape[0]} samples x {X.shape[1]} genes -> {out}")
    return report
