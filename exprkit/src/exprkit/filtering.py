"""Unsupervised gene filtering as a scikit-learn transformer.

The filter is fitted inside every training fold (it is the first step of each
model pipeline), so the genes are always chosen from training samples only.
Choosing genes on all samples before cross validation biases the error
estimate (Ambroise and McLachlan 2002, PNAS 99:6562). The steps below do not
use the labels, so that bias would be small, but fitting inside the folds
removes the question.

Input: samples in rows, genes in columns.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def mode_fraction(values: np.ndarray) -> np.ndarray:
    """Fraction of samples taken by the most frequent value in each column
    (NaN never counts as equal to NaN)."""
    n, p = values.shape
    if n < 2:
        return np.ones(p)
    out = np.empty(p)
    for start in range(0, p, 2000):              # column blocks keep memory modest for 30,000 genes
        s = np.sort(values[:, start:start + 2000], axis=0)
        eq = (s[1:] == s[:-1]).astype(np.int32)
        c = np.cumsum(eq, axis=0)
        reset = np.maximum.accumulate(np.where(eq == 0, c, 0), axis=0)
        out[start:start + 2000] = ((c - reset).max(axis=0) + 1) / n
    return out


def duplicate_columns(values: np.ndarray) -> np.ndarray:
    """Boolean mask, True for a column identical to an earlier column."""
    seen: dict[bytes, int] = {}
    dup = np.zeros(values.shape[1], dtype=bool)
    cols = np.ascontiguousarray(values.T)
    for j in range(cols.shape[0]):
        key = cols[j].tobytes()
        if key in seen:
            dup[j] = True
        else:
            seen[key] = j
    return dup


class ExpressionFilter(BaseEstimator, TransformerMixin):
    """Drop uninformative genes, in this order:

    1. more than ``max_missing_fraction`` of values missing
    2. quasi constant: the most frequent value covers at least ``max_mode_fraction`` of samples
    3. exact duplicates of an earlier gene
    4. mean below ``min_mean`` (optional; on the scale of the data)
    5. fewer than ``min_samples_above`` samples above ``expressed_value`` (optional)
    6. keep the ``top_k_variance`` most variable remaining genes (optional)
    """

    def __init__(self, max_missing_fraction: float = 0.5, max_mode_fraction: float = 0.95,
                 drop_duplicates: bool = True, min_mean: float | None = None,
                 expressed_value: float | None = None, min_samples_above: int = 1,
                 top_k_variance: int | None = None):
        self.max_missing_fraction = max_missing_fraction
        self.max_mode_fraction = max_mode_fraction
        self.drop_duplicates = drop_duplicates
        self.min_mean = min_mean
        self.expressed_value = expressed_value
        self.min_samples_above = min_samples_above
        self.top_k_variance = top_k_variance

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        self.n_features_in_ = df.shape[1]
        self.feature_names_in_ = np.asarray(df.columns.astype(str), dtype=object)
        v = df.to_numpy(dtype=np.float64)
        keep = np.ones(v.shape[1], dtype=bool)
        rep = {"input": int(v.shape[1])}
        keep &= np.isnan(v).mean(axis=0) <= self.max_missing_fraction
        rep["after_missing"] = int(keep.sum())
        if self.max_mode_fraction is not None and self.max_mode_fraction < 1:
            keep &= mode_fraction(v) < self.max_mode_fraction
        rep["after_constant"] = int(keep.sum())
        if self.drop_duplicates:
            idx = np.flatnonzero(keep)
            keep[idx[duplicate_columns(v[:, idx])]] = False
        rep["after_duplicates"] = int(keep.sum())
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.min_mean is not None:
                keep &= np.nan_to_num(np.nanmean(v, axis=0), nan=-np.inf) >= self.min_mean
            if self.expressed_value is not None:
                keep &= (v > self.expressed_value).sum(axis=0) >= self.min_samples_above
            rep["after_expression_thresholds"] = int(keep.sum())
            if self.top_k_variance is not None and keep.sum() > self.top_k_variance:
                var = np.nan_to_num(np.nanvar(v, axis=0), nan=-np.inf)
                var[~keep] = -np.inf
                top = np.argsort(var)[::-1][: self.top_k_variance]
                keep = np.zeros_like(keep)
                keep[top] = True
        rep["kept"] = int(keep.sum())
        if keep.sum() == 0:
            raise ValueError(f"ExpressionFilter removed every gene: {rep}. Thresholds do not suit this data scale.")
        self.support_ = keep
        self.report_ = rep
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.support_]
        return np.asarray(X)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.support_]
