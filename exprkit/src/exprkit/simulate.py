"""Simulated data with known answers, used only by the test suite.

Nothing here is real patient data. The fake files copy the *layout* of real
GEO downloads (series matrix, platform table, a genes by samples CSV) so the
readers and commands are tested end to end against answers that are known in
advance. Real analyses start from real files with ``exprkit prepare``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def simulate_expression(n_cancer=60, n_normal=10, n_genes=500, n_signal=20, effect=1.0,
                        mean=7.0, sd=1.5, seed=0, prefix="G", gene_seed=None):
    """Log scale expression, samples x genes. The first ``n_signal`` genes are
    shifted by ``effect`` (in gene standard deviations) in cancer samples.
    ``gene_seed`` fixes the gene means separately from the sample noise, so two
    calls with the same ``gene_seed`` mimic two cohorts on the same platform."""
    gene_means = np.random.default_rng(seed if gene_seed is None else gene_seed).normal(mean, sd, n_genes)
    rng = np.random.default_rng(seed + 1000 if gene_seed is not None else seed)
    if gene_seed is None:
        rng.normal(mean, sd, n_genes)
    n = n_cancer + n_normal
    X = rng.normal(0, 1, (n, n_genes)) + gene_means
    y = np.r_[np.ones(n_cancer, int), np.zeros(n_normal, int)]
    X[y == 1, :n_signal] += effect
    genes = [f"{prefix}{i:05d}" for i in range(n_genes)]
    return pd.DataFrame(X, columns=genes, index=[f"S{i:04d}" for i in range(n)]), y


def simulate_survival(X: np.ndarray, beta: np.ndarray, baseline_rate=1e-3, censor_max=4000, seed=0):
    """Exponential event times with log hazard ``X @ beta`` and uniform censoring.

    Returns time, event and the true log hazard (the ideal risk score).
    """
    rng = np.random.default_rng(seed)
    log_h = X @ beta
    t_event = rng.exponential(1.0 / (baseline_rate * np.exp(log_h)))
    t_cens = rng.uniform(100, censor_max, len(log_h))
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    return np.round(time, 3) + 1.0, event, log_h


def _series_matrix(path: Path, meta_lines: list[tuple[str, list[str]]], table: pd.DataFrame | None):
    with open(path, "w") as fh:
        fh.write('!Series_title\t"SIMULATED DATA, not real patients"\n')
        for key, vals in meta_lines:
            fh.write(key + "\t" + "\t".join(f'"{v}"' for v in vals) + "\n")
        fh.write("!series_matrix_table_begin\n")
        if table is not None:
            fh.write('"ID_REF"\t' + "\t".join(f'"{c}"' for c in table.columns) + "\n")
            for pid, row in table.iterrows():
                fh.write(f'"{pid}"\t' + "\t".join(f"{v:.4f}" for v in row.values) + "\n")
        fh.write("!series_matrix_table_end\n")


def write_fake_geo(outdir: str | Path, seed: int = 0) -> dict:
    """Write a small fake copy of the two GEO studies used in the project.

    * ``SIM_A``: a microarray study laid out like GSE22820, with probe IDs, a
      platform table, 60 tumour and 10 normal samples, labels in
      ``!Sample_source_name_ch1``, and the normal samples placed in the
      *middle* of the columns so that position based labelling would fail.
    * ``SIM_B``: an RNA sequencing study laid out like GSE96058, gene symbols
      by samples in a CSV, two series matrices (two platforms) with survival
      characteristics, some replicate samples, and some missing events.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_genes = 400

    # Study A ----------------------------------------------------------------
    XA, yA = simulate_expression(60, 10, n_genes, n_signal=25, effect=1.2, seed=seed, gene_seed=99)
    order = rng.permutation(len(yA))           # shuffle so normals are not last
    XA, yA = XA.iloc[order], yA[order]
    gsm_a = [f"GSMA{i:04d}" for i in range(len(yA))]
    genes = list(XA.columns)
    # Platform: 1 to 2 probes per gene, plus unannotated and ambiguous probes.
    probes, gpl_rows = [], []
    for g in genes:
        for k in range(rng.integers(1, 3)):
            pid = f"P_{g}_{k}"
            probes.append((pid, g))
            gpl_rows.append((pid, g))
    gpl_rows += [("P_CTRL_1", ""), ("P_CTRL_2", ""), ("P_AMBIG_1", f"{genes[0]} /// {genes[1]}")]
    probe_tab = pd.DataFrame(
        {gsm: [XA.iloc[j][g] + rng.normal(0, 0.1) for _, g in probes] for j, gsm in enumerate(gsm_a)},
        index=[p for p, _ in probes])
    extra = pd.DataFrame(rng.normal(5, 1, (3, len(gsm_a))), index=["P_CTRL_1", "P_CTRL_2", "P_AMBIG_1"], columns=gsm_a)
    probe_tab = pd.concat([probe_tab, extra])
    src = ["normal breast tissue" if v == 0 else "primary breast tumor" for v in yA]
    _series_matrix(out / "SIM_A_series_matrix.txt",
                   [("!Sample_title", [f"A_{i}" for i in range(len(yA))]),
                    ("!Sample_geo_accession", gsm_a),
                    ("!Sample_source_name_ch1", src)], probe_tab)
    with open(out / "SIM_A_platform.txt", "w") as fh:
        fh.write("# ID = probe identifier\n# GENE_SYMBOL = gene symbol\n")
        fh.write("ID\tGENE_SYMBOL\tSEQUENCE\n")
        for pid, sym in gpl_rows:
            fh.write(f"{pid}\t{sym}\tACGT#ACGT\n")

    # Study B ----------------------------------------------------------------
    nB = 600
    XB, _ = simulate_expression(nB, 0, n_genes, n_signal=0, mean=3.0, sd=2.0, seed=seed + 1)
    XB += 0.8  # a study shift, as between platforms
    beta = np.zeros(n_genes)
    beta[:10] = rng.normal(0, 0.35, 10)
    age = rng.normal(62, 12, nB).round()
    size = np.abs(rng.normal(22, 10, nB)).round(1)
    Xc = (XB.to_numpy() - XB.to_numpy().mean(0)) / XB.to_numpy().std(0)
    # Clinical variables carry part of the risk, genes carry the rest, so a
    # gene model should beat the clinical model by a known margin.
    X_all = np.c_[Xc, (age - 62) / 12, (size - 22) / 10]
    beta_all = np.r_[beta, 0.3, 0.4]
    time, event, _ = simulate_survival(X_all, beta_all, seed=seed + 2)
    titles = [f"F{i}" for i in range(nB)]
    gsm_b = [f"GSMB{i:04d}" for i in range(nB)]
    ev_text = [str(int(v)) for v in event]
    for i in rng.choice(nB, 15, replace=False):  # missing outcomes
        ev_text[i] = "NA"
    # 20 technical replicates of existing tumours
    rep_src = rng.choice(nB, 20, replace=False)
    rep_titles = [f"F{i}repl" for i in rep_src]
    rep_gsm = [f"GSMB9{k:03d}" for k in range(20)]
    XB_rep = XB.iloc[rep_src].to_numpy() + rng.normal(0, 0.05, (20, n_genes))
    exprB = pd.concat([XB.set_axis(titles), pd.DataFrame(XB_rep, columns=genes, index=rep_titles)])
    exprB.T.rename_axis("gene").to_csv(out / "SIM_B_gene_expression.csv")

    all_titles = titles + rep_titles
    all_gsm = gsm_b + rep_gsm
    all_time = list(time) + [time[i] for i in rep_src]
    all_ev = ev_text + [ev_text[i] for i in rep_src]
    all_age = list(age) + [age[i] for i in rep_src]
    all_size = list(size) + [size[i] for i in rep_src]
    half = len(all_titles) // 2
    for part, sl in (("GPL1", slice(0, half)), ("GPL2", slice(half, None))):
        _series_matrix(out / f"SIM_B-{part}_series_matrix.txt", [
            ("!Sample_title", all_titles[sl]),
            ("!Sample_geo_accession", all_gsm[sl]),
            ("!Sample_characteristics_ch1", [f"age at diagnosis: {int(a)}" for a in all_age[sl]]),
            ("!Sample_characteristics_ch1", [f"tumor size: {s}" for s in all_size[sl]]),
            ("!Sample_characteristics_ch1", [f"overall survival days: {t:.0f}" for t in all_time[sl]]),
            ("!Sample_characteristics_ch1", [f"overall survival event: {e}" for e in all_ev[sl]]),
        ], None)
    return {"study_a_normals": int((yA == 0).sum()), "study_a_tumours": int((yA == 1).sum()),
            "study_b_samples": len(all_titles), "study_b_replicates": 20, "study_b_missing_events": 15}


def write_fake_external(outdir: str | Path, seed: int = 5) -> dict:
    """A new labelled cohort on the same simulated platform as SIM_A, with
    gene symbols already in place: ``SIM_C_expression.csv`` (genes by samples,
    5 genes missing) and ``SIM_C_samples.csv`` (sample_id, tissue)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = simulate_expression(40, 12, 400, n_signal=25, effect=1.2, seed=seed, gene_seed=99)
    X.index = [f"C{i:03d}" for i in range(len(y))]
    X = X.drop(columns=X.columns[-5:])
    X.T.rename_axis("symbol").to_csv(out / "SIM_C_expression.csv")
    pd.DataFrame({"sample_id": X.index, "tissue": np.where(y == 1, "Tumor tissue", "Normal tissue")}) \
        .to_csv(out / "SIM_C_samples.csv", index=False)
    return {"n": len(y), "normals": int((y == 0).sum())}
