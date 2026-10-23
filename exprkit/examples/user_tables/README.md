# Ready-made tables

Many analyses start from tables that were already combined, labelled and split: samples in rows, genes in columns, plus a label or survival columns. This example reads such tables, checks them, and runs honest models on them. Put your files in `data/user/`. That folder is not tracked by git.

| File | Content |
|---|---|
| `train_data.csv`, `val_data.csv`, `test_data.csv` | 79 genes and a 0/1 `Label` column; no sample IDs |
| `merged_expression_survival.csv` | 216 genes plus `Sample_title`, `Sample ID`, age, tumour size, overall survival days and event |

## Steps

```bash
exprkit prepare examples/user_tables/prepare_classification_tables.yaml   # duplicates: drop
exprkit prepare examples/user_tables/prepare_survival_table.yaml
python examples/user_tables/assign_study.py          # needs data/processed/gse96058 (breast cancer example)

exprkit train examples/user_tables/train_pooled_as_provided.yaml
exprkit train examples/user_tables/train_confounding_test.yaml
exprkit evaluate --model results/user_tables/confounding_test/model.joblib \
    --data data/processed/user_classification --where confounding_design=held_out \
    --out results/user_tables/confounding_test/gse22820_tumours
exprkit train examples/user_tables/train_within_gse22820.yaml
exprkit train examples/user_tables/train_survival_table_gse96058.yaml
exprkit train examples/user_tables/train_survival_table_all_cohorts.yaml
```

## What the checks found

* **Copied normals.** The 2,868 normal rows in `train_data.csv` are 8 distinct profiles copied 344 to 377 times. The validation and test files have 1 normal each.
* **All normals from one study.** Every normal comes from GSE22820, while 3,409 of 3,585 tumours come from GSE96058. A model trained on these tables called 173 of 176 unseen GSE22820 tumours "normal".
* **Imputed tumour sizes.** In the survival table, tumour size for 734 rows from three other series is exactly 12.50 + 0.118 × age, so it was filled in from age. The prepare config sets those values to missing.

The full numbers are in sections 9 and 10 of [docs/case_study_breast_cancer.md](../../docs/case_study_breast_cancer.md).

## Using the same approach for your own tables

* List every non gene column under `metadata_columns`.
* Map coded labels with `map: {0: normal, 1: cancer}`.
* Keep `duplicates: warn` or `drop`.
* Never balance classes by copying rows; use `balance: undersample` or the default class weights in train.
