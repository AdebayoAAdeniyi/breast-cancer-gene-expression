# Breast cancer example

A complete exprkit workflow on two public GEO studies. The decisions and results are explained in [docs/case_study_breast_cancer.md](../../docs/case_study_breast_cancer.md).

## Files needed in `data/raw/`

| File | Study | Content |
|---|---|---|
| `GSE22820_series_matrix.txt` | GSE22820 | Microarray values and sample descriptions (176 tumour, 10 normal) |
| `GPL6480-9577.txt` | GPL6480 | Probe ID to gene symbol table ("Download full table" on the GPL page) |
| `GSE96058-GPL11154_series_matrix.txt` | GSE96058 | Clinical data for 3,069 samples (HiSeq 2000) |
| `GSE96058-GPL18573_series_matrix.txt` | GSE96058 | Clinical data for 340 samples (NextSeq 500) |
| `GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz` | GSE96058 | RNA sequencing values (supplementary file, 592 MB) |
| `GSE15852_series_matrix.txt`, `GSE42568_series_matrix.txt`, `GSE70947_series_matrix.txt` | multi study example | studies with both normal and tumour tissue |
| `GPL96.annot.gz`, `GPL570.annot.gz`, `GPL13607_table.txt` | multi study example | probe to gene tables for those platforms |

`download_geo.sh` fetches every file except the GPL6480 table (save it from the GPL6480 page with "Download full table") and writes `SHA256SUMS.txt`. `download_geo.R` uses Bioconductor GEOquery instead and also saves the GPL6480 and GPL13607 tables. The configs read series matrices uncompressed.

## Configs

| Config | What it does | Needs the expression CSV |
|---|---|---|
| `prepare_gse22820.yaml` | maps 41,000 probes to 19,595 genes, applies log2, takes labels from `source_name_ch1` with a 10/176 count check | no |
| `train_cancer_vs_normal.yaml` | logistic regression, 20 × 5 fold nested cross validation, 200 permutations | no |
| `prepare_gse96058_clinical.yaml` | clinical columns only, 136 replicate check | no |
| `train_survival_clinical_age_size.yaml` | Cox on age and tumour size | no |
| `train_survival_clinical_full.yaml` | Cox on age, size, node status, grade, ER, HER2 | no |
| `prepare_gse96058.yaml` | expression and clinical data | yes |
| `train_survival.yaml` | elastic net Cox on genes and clinical factors, compared with clinical only | yes |
| `train_pam50.yaml` | five class PAM50 subtype | yes |
| `train_survival_79genes.yaml` | survival with a supplied gene list (`gene_lists/`) + clinical factors | yes |
| `select_genes_survival.yaml` | survival gene list from the two GSE96058 platforms, with evidence table | yes |
| `train_survival_infold_selection.yaml` | 50 genes chosen inside every training fold + clinical factors | yes |
| `prepare_multistudy_tumour_normal.yaml` | GSE22820, GSE15852, GSE42568, GSE70947 on shared genes, per sample ranks | no (needs the three extra series and GPL96, GPL570, GPL13607 tables) |
| `train_multistudy_*.yaml` | leave one study out (plain, with selection and undersampling, with the 79 genes) and pooled CV for contrast | no |
| `select_genes_tumour_normal.yaml` | tumour versus normal genes consistent across the four studies | no |

## Run

```bash
bash examples/breast_cancer/run.sh
```

The script skips the expression based GSE96058 steps when that file is absent.

## What not to do with these data

* Do not train tumour versus normal on pooled studies. All normal samples come from GSE22820, so a pooled model can score well just by recognising the study.
* Do not apply the GSE22820 classifier to GSE96058 and read the output as accuracy. GSE96058 has no normal tissue, uses a different platform, and was processed differently. `predict` flags the scale difference.
* Do not use GSE118389 as a validation cohort for these models. It is single cell RNA sequencing of 1,534 cells from 6 triple negative tumours, not bulk tumour profiles.
