"""Value transformations and scale checks.

Models in exprkit assume log scale expression (microarray log2 intensities,
log2 FPKM, TPM or CPM). Raw counts or raw intensities must be transformed in
``prepare``. The transform chosen is stored with the dataset and with every
model trained on it, so ``predict`` can warn when new data are on another scale.

Available transforms
--------------------
none     values are already on a log scale
log2     log2(x); fails if any value is zero or negative
log2p1   log2(x + 1); for FPKM, TPM or intensities containing zeros
log2cpm  log2(counts per million + 1), library size = column sum per sample.
         Plain CPM, without TMM or other composition normalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRANSFORMS = ("none", "log2", "log2p1", "log2cpm")


def apply_transform(X: pd.DataFrame, kind: str) -> pd.DataFrame:
    """``X`` has samples in rows."""
    if kind not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {kind!r}")
    if kind == "none":
        return X
    v = X.to_numpy(dtype=np.float64)
    if np.nanmin(v) < 0:
        raise ValueError(f"transform '{kind}' received negative values; the data are probably already "
                         "on a log scale, so use transform: none")
    if kind == "log2":
        if np.nanmin(v) <= 0:
            raise ValueError("transform 'log2' received zeros; use log2p1")
        out = np.log2(v)
    elif kind == "log2p1":
        out = np.log2(v + 1.0)
    else:
        lib = np.nansum(v, axis=1, keepdims=True)
        if (lib <= 0).any():
            raise ValueError("log2cpm: some samples have a total count of zero")
        out = np.log2(v / lib * 1e6 + 1.0)
    return pd.DataFrame(out.astype(np.float32), index=X.index, columns=X.columns)


def scale_summary(X: pd.DataFrame) -> dict:
    v = X.to_numpy(dtype=np.float64).ravel()
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n_values": 0}
    q = [0.01, 0.25, 0.5, 0.75, 0.99]
    return {
        "n_samples": int(X.shape[0]), "n_genes": int(X.shape[1]),
        "min": float(v.min()), "max": float(v.max()),
        "quantiles": dict(zip(map(str, q), np.quantile(v, q).round(3).tolist())),
        "fraction_missing": float(X.isna().to_numpy().mean()),
        "looks_log_scale": bool(v.max() < 50),
        "looks_integer_counts": bool(v.max() > 100 and np.allclose(v[:100000], np.round(v[:100000]))),
    }


def scale_warnings(X: pd.DataFrame, kind: str) -> list[str]:
    s = scale_summary(X)
    w = []
    if s.get("n_values") == 0:
        return ["no values"]
    if kind == "none" and not s["looks_log_scale"]:
        w.append(f"maximum value {s['max']:.1f} suggests the data are not log transformed; "
                 "consider transform: log2p1 (or log2cpm for counts)")
    if kind != "none" and s["looks_log_scale"]:
        w.append(f"maximum value {s['max']:.1f} suggests the data are already log transformed; "
                 f"transform: {kind} would log them twice")
    if s["looks_integer_counts"] and kind not in ("log2cpm",):
        w.append("values look like raw integer counts; consider transform: log2cpm")
    return w


def per_sample_rank(X: pd.DataFrame) -> pd.DataFrame:
    """Rank genes within each sample and map ranks to normal scores.

    Value = inverse normal of (rank - 0.5) / n, where n is the number of
    non missing genes in that sample. Each sample is transformed on its own,
    so no information passes between samples (or between train and test),
    and platform specific scales (log ratios, log intensities, log FPKM)
    become comparable as long as the same genes are used. Ties get their
    average rank; missing values stay missing.
    """
    from scipy import stats
    r = X.rank(axis=1, method="average", na_option="keep")
    n = X.notna().sum(axis=1).replace(0, np.nan)
    q = r.sub(0.5).div(n, axis=0)
    return pd.DataFrame(stats.norm.ppf(q.to_numpy(dtype=float)).astype(np.float32), index=X.index, columns=X.columns)


NORMALISATIONS = ("none", "per_sample_rank")


def apply_normalisation(X: pd.DataFrame, kind: str | None) -> pd.DataFrame:
    if kind in (None, "none"):
        return X
    if kind == "per_sample_rank":
        return per_sample_rank(X)
    raise ValueError(f"normalise must be one of {NORMALISATIONS}")
