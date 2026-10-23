#!/usr/bin/env bash
# Breast cancer worked example. Run from the repository root with the raw files
# in data/raw (see examples/breast_cancer/README.md).
set -euo pipefail
EX=examples/breast_cancer
mkdir -p results

# 1. Look at the inputs
exprkit inspect data/raw/GSE22820_series_matrix.txt          > results/inspect_gse22820.txt
exprkit inspect data/raw/GSE96058-GPL11154_series_matrix.txt > results/inspect_gse96058.txt

# 2. Cancer versus normal tissue within GSE22820
exprkit prepare $EX/prepare_gse22820.yaml
exprkit train   $EX/train_cancer_vs_normal.yaml

# 3. Overall survival from clinical factors only (no expression file needed)
exprkit prepare $EX/prepare_gse96058_clinical.yaml
exprkit train   $EX/train_survival_clinical_age_size.yaml
exprkit train   $EX/train_survival_clinical_full.yaml

# 4. Steps that need the GSE96058 expression file
EXPR=data/raw/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz
if [ -f "$EXPR" ]; then
  exprkit inspect "$EXPR" > results/inspect_gse96058_expression.txt
  exprkit prepare $EX/prepare_gse96058.yaml
  exprkit train   $EX/train_survival.yaml
  exprkit train   $EX/train_pam50.yaml
  # Can the two studies be told apart using tumours only?
  exprkit combine --data data/processed/gse22820 --data data/processed/gse96058 --out data/processed/combined
  exprkit batch-check --data data/processed/combined --batch-column dataset --where label=cancer \
      --out results/batch_check_gse22820_vs_gse96058.json
  # Survival with a supplied gene list, with a list built by select-genes, and with selection inside the folds
  exprkit train   $EX/train_survival_79genes.yaml
  exprkit select-genes $EX/select_genes_survival.yaml
  exprkit train   $EX/train_survival_infold_selection.yaml
else
  echo "Skipping expression based GSE96058 steps: $EXPR not found"
fi

# 5. Tumour versus normal across four studies that each have both classes
if [ -f data/raw/GSE70947_series_matrix.txt ]; then
  exprkit prepare $EX/prepare_multistudy_tumour_normal.yaml
  for v in plain select_balance genes79 pooled_cv; do exprkit train $EX/train_multistudy_$v.yaml; done
  exprkit batch-check --data data/processed/multistudy_tumour_normal --where label=normal \
      --out results/batch_check_multistudy_normal.json
  exprkit select-genes $EX/select_genes_tumour_normal.yaml
fi
