# Gene lists

Plain text files with one gene symbol per line. Use one in a train or select-genes config:

```yaml
genes:
  gene_list: gene_lists/breast_cancer_79_selected.txt
```

Genes missing from your data are reported in `summary.json` (`gene_list.missing`). Symbols are matched exactly, after upper casing.

## Lists in this folder

| File | Genes | How it was made |
|---|---|---|
| `breast_cancer_216_candidates.txt` | 216 | Genes common to several breast cancer expression files, kept after filtering (the candidate set) |
| `breast_cancer_79_selected.txt` | 79 | Chosen from a larger candidate set by occurrence across files, importance in models, and removal of highly correlated genes. 17 of the 79 are not among the 216 candidates. |
| `breast_tumour_normal_4studies_50.txt` | 50 | Made by `exprkit select-genes` with `examples/breast_cancer/select_genes_tumour_normal.yaml`: present in GSE22820, GSE15852, GSE42568 and GSE70947, same direction in all four, stable, correlation at most 0.8 |
| `breast_cancer_survival_gse96058_50.txt` | 50 | Made by `exprkit select-genes` with `examples/breast_cancer/select_genes_survival.yaml`; the evidence for every gene is in `results/gene_selection_gse96058_survival/gene_selection.csv` |

## Using a list honestly

A list chosen using some samples carries information about those samples. A model trained and tested on the same samples with that list scores higher than it would on new samples. The higher the number of genes screened, the larger the bias: with 2,000 genes, 60 samples and no real signal at all, choosing the top 20 genes first and then cross validating gives an AUC of 0.94, while choosing them inside each fold gives 0.52 (`tests/test_selection.py::test_selection_outside_cv_is_optimistic`).

Two ways to get an honest estimate:

1. Evaluate the list on samples that played no part in choosing it (`exprkit evaluate` on an independent cohort).
2. Let `train` repeat the selection inside each training fold with `genes.select`. The result measures the whole procedure, including the choice of genes.

Every list here was chosen using samples from the studies in the example, so results on those studies with these lists are optimistic. Treat them as descriptive, and validate the lists elsewhere.
