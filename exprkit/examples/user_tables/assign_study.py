"""Recover which GEO study each row of the ready-made classification tables
came from, by correlating each row's 79 gene profile with GSE96058 profiles
(standardised per gene, as in the tables).

Rows whose best correlation with any GSE96058 sample is below 0.7 are assigned
to GSE22820. On the files checked (22 October 2026) the split is sharp: 186
rows have a best correlation of at most 0.62 (176 tumours and the 10 normals,
matching GSE22820's 176 + 10), and every other row has at least 0.78.

Usage (after preparing both GEO datasets and the user tables):
    python examples/user_tables/assign_study.py
It adds a 'study' column to data/processed/user_classification/samples.csv.
"""
import numpy as np
import pandas as pd

from exprkit.io import load_dataset, save_dataset

X, S, report = load_dataset("data/processed/user_classification")
B, _, _ = load_dataset("data/processed/gse96058")
genes = [g for g in X.columns if g in B.columns]


def row_normalise(M):
    M = M - M.mean(axis=1).to_numpy()[:, None]
    return M.div(np.sqrt((M ** 2).sum(axis=1)), axis=0)


ZB = B[genes].astype(float)
ZB = (ZB - ZB.mean()) / ZB.std(ddof=0)
best = np.empty(len(X))
U = row_normalise(X[genes].astype(float)).to_numpy()
V = row_normalise(ZB).to_numpy().T
for s in range(0, len(U), 500):                  # in blocks to limit memory
    best[s:s + 500] = (U[s:s + 500] @ V).max(axis=1)
S["best_r_with_GSE96058"] = best.round(4)
S["study"] = np.where(best < 0.7, "GSE22820", "GSE96058")
# For the confounding test: normals (all GSE22820) against GSE96058 tumours only,
# leaving the GSE22820 tumours out of training so they can be scored afterwards.
S["confounding_design"] = np.where((S["label"] == "cancer") & (S["study"] == "GSE22820"), "held_out", "train")
print(pd.crosstab(S["label"], S["study"]))
save_dataset("data/processed/user_classification", X, S, {**report, "study_assignment": "examples/user_tables/assign_study.py"})
