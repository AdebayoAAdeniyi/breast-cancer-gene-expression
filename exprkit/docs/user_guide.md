# exprkit user guide

This guide covers the workflow, every config option, the output files, and how to read the results. Relative paths in configs are read from the folder you run `exprkit` in.

## 1. Workflow

| Step | Command | Input | Output |
|---|---|---|---|
| Look | `exprkit inspect FILE` | any input file | console: sample IDs, metadata fields and values, value scale |
| Prepare | `exprkit prepare prepare.yaml` | raw files | prepared dataset folder |
| Combine (optional) | `exprkit combine --data A --data B --out AB` | prepared datasets | one prepared dataset, `dataset` column per sample |
| Check batches (optional) | `exprkit batch-check --data AB --where label=tumour` | prepared dataset | AUC for telling two batches apart |
| Select genes (optional) | `exprkit select-genes select.yaml` | one or more prepared datasets | `genes.txt` and the evidence table |
| Train | `exprkit train train.yaml` | prepared dataset | evaluation reports and `model.joblib` |
| Predict | `exprkit predict --model M --data D --out p.csv` | saved model and new samples | predictions, input report |
| Validate | `exprkit evaluate --model M --data D --out dir` | saved model and new labelled samples | external metrics with intervals |

`exprkit template prepare`, `exprkit template train` and `exprkit template select-genes` print commented templates that list every option.

## 2. Inspect first

```bash
exprkit inspect data/raw/GSE22820_series_matrix.txt.gz
exprkit inspect data/raw/counts.csv --orientation genes_by_samples
exprkit inspect data/raw/GPL6480-9577.txt --format annotation --id-column ID --symbol-column GENE_SYMBOL
```

For a series matrix, inspect prints every `!Sample_` field with its distinct values. Characteristic lines appear as `char:<name>`, for example `char:overall survival days`. Use these exact names in the prepare config.

For a table, it prints the first sample and feature names and a scale summary. A maximum below about 50 means the values are on a log scale. Large integers mean raw counts, which need `transform: log2cpm`.

## 3. Prepare

### 3.1 Sources

Each entry under `sources` is one study. Its parts:

**`expression`**
* `format: geo_series_matrix`: the values are read from the matrix table. Some series store their values in a supplementary file and leave this table empty; for those, prepare stops and tells you to use `format: table` for the supplementary file.
* `format: table`: a CSV or TSV file, optionally gzipped. Set `orientation` to `genes_by_samples` (the default) or `samples_by_genes`. `index_column` names the column that holds the IDs. Non numeric columns, such as a description column, are dropped and listed.

**Ready-made tables with labels inside.** Many people keep data as one table: samples in rows, genes in columns, plus columns such as `Label`, `Age` or survival time. Read it with `orientation: samples_by_genes`, and list the non gene columns under `metadata_columns`. They then become metadata you can derive columns from. `index_column` names the sample ID column. Use `index_column: null` when there is none; rows are then named `<file>_1`, `<file>_2`, and so on.

```yaml
expression:
  path: data/user/train_data.csv
  format: table
  orientation: samples_by_genes
  index_column: null
  metadata_columns: [Label]
columns:
  label: {type: category, from: Label, map: {0: normal, 1: cancer}}
```

Omit `expression` to build a clinical only dataset from metadata. It has zero gene columns, and you train on it with `genes: {use: false}`. This is useful for setting the clinical baseline before any expression data are ready.

**`metadata`**
* `geo_series_matrix`: one or more series matrices. A series run on two platforms has one per platform; list both.
* `sample_sheet`: your own CSV with `path` and `id_column`.
* Without this block, a series matrix supplies its own metadata.

**`match_expression_on`**: use this when the expression file names samples differently from the metadata IDs, for example sample titles versus GSM accessions. Give the metadata field whose values equal the expression sample names. Samples that cannot be linked are counted and listed in the report.

**`transform`**: one of `none`, `log2`, `log2p1` or `log2cpm`. The transform is applied before ID mapping, so library sizes use all features and probes are averaged on the log scale. Prepare warns when the choice looks wrong for the data, for example raw counts with `none`, or `log2p1` on values that are already logged.

**`id_mapping`**: an annotation table (`path`, `id_column`, `symbol_column`). This is usually the GEO platform "full table".
* IDs without a symbol are dropped and listed.
* IDs annotated to several genes (`A /// B`) are dropped when `drop_ambiguous: true`.
* Several IDs for one gene are averaged.
* Without `id_mapping`, the existing feature names are treated as symbols. They are upper cased, and duplicates are averaged.

**`columns`**: the columns to create in `samples.csv`.

| type | what it does | extra keys |
|---|---|---|
| `category` | class from regular expression rules; every sample must match exactly one rule | `rules`, `expected` (counts) |
| `category` without `rules` | the text values themselves become the classes | `expected` |
| `number` | first number in the text (`"22.5 mm"` becomes 22.5) | |
| `event` | 1 for yes, dead, deceased, event or true; 0 for no, alive, censored or false; anything else is missing | `true_values`, `false_values` |
| `text` | kept as text (useful for filters) | |
| `group` | patient ID; `strip_regex` removes a replicate suffix | `expected_groups_with_replicates` |
| `value` | the same constant for every sample | |
| `category` with `map` | coded values translated, e.g. `{0: normal, 1: cancer}`; 1, 1.0 and "1" all match | `map`, `expected` |

`from: _id` uses the sample ID itself, for example to tell cohorts apart by accession prefix. `missing_when: {field: F, pattern: R}` sets a column to missing wherever metadata field F (or `_id`) matches the regular expression R. Use it for values you know are unusable, such as numbers imputed upstream.

For every type, text such as `NA`, `N/A`, `unknown` or an empty string means missing (override with `na_values`). A missing category stays missing; it never becomes a class called "NA".

The `expected` checks are the most important lines in a config. They tie your rules to numbers you know independently, such as the sample counts on the GEO record. If the counts differ, prepare stops.

### 3.2 Several sources

With more than one source, `combine: inner` keeps the genes present in every study, and `combine: outer` keeps all genes, leaving missing values where a study lacks a gene. The `dataset` column records each sample's source.

### 3.3 Duplicates and normalisation

`duplicates` checks for identical expression profiles under different sample IDs. These usually mean copies made to balance classes, or samples deposited twice:
* `warn` (the default) reports them.
* `drop` keeps the first copy.
* `group` keeps all copies and writes a `duplicate_group` column to use as `group_column`, so copies never fall on both sides of a split.

`normalise: per_sample_rank` replaces each sample's values by the normal scores of their ranks within that sample. Different platforms (log ratios, log intensities, log FPKM) then become comparable, without passing any information between samples. A saved model remembers the genes it was ranked over and applies the same step to new samples, so data for `predict` and `evaluate` must be prepared *without* `normalise`. `exprkit combine --normalise per_sample_rank` does the same after combining prepared datasets.

### 3.4 Output

```
<output>/expression.parquet   samples x genes, float32
<output>/samples.csv          derived columns plus 'dataset'
<output>/report.json          per source: shapes, linking, transform, scale before and after,
                              mapping counts with examples of dropped IDs, column summaries
```

## 4. Train

### 4.1 Rows

* `where: {column: [values]}` keeps only matching rows.
* Rows with a missing target are excluded and counted. For survival, that means a missing or non positive time, or an event that is not 0 or 1.
* `group_column` keeps rows that share a value in the same fold.
* `one_per_group: true` keeps only the first row of each group, for example to drop technical replicates.

### 4.2 Features

* `genes.use`: set to false for a covariates only model.
* `genes.gene_list`: a text file with one symbol per line, to restrict the model to a signature. Genes not found are reported.
* `genes.filter`: fitted inside each training fold. The steps, in order:
  1. drop genes with too many missing values (`max_missing_fraction`)
  2. drop quasi constant genes (`max_mode_fraction`)
  3. drop exact duplicates
  4. apply optional thresholds (`min_mean`, `expressed_value` with `min_samples_above`)
  5. keep the `top_k_variance` most variable genes
* `genes.select`: choose genes inside each training fold, after the filter: `{n_genes: 50, max_correlation: 0.7}`. Genes are ranked by univariate association with the target. For two classes this is the Welch t, for more classes the ANOVA F, for survival the Cox score test, and for regression the correlation. A gene is skipped when it correlates above `max_correlation` with a better ranked gene. Because this happens inside the folds, the evaluation includes the effect of choosing genes.
* `covariates.numeric` and `covariates.categorical`: columns from `samples.csv`. Numeric covariates are median imputed and standardised. Categorical covariates are imputed with the most frequent value and one hot encoded.

### 4.3 Models and tuning

| task | `model.type` | grid keys |
|---|---|---|
| classification | `logistic` (L2, class weighted) | `C` |
| | `elasticnet_logistic` | `C`, `l1_ratio` |
| | `random_forest` (class weighted) | `n_estimators`, `max_depth`, `min_samples_leaf` |
| | `mlp` | `alpha`, `hidden_layer_sizes` |
| regression | `ridge` | `alpha` |
| | `elasticnet` | `alpha`, `l1_ratio` |
| | `random_forest` | `max_depth`, `min_samples_leaf` |
| survival | `cox` (ridge penalised) | `alpha` |
| | `coxnet` (elastic net) | `alpha`, `l1_ratio` |
| | `deepsurv` (needs `pip install -e ".[deep]"`) | `hidden`, `dropout`, `lr`, `weight_decay` |

`model.params` sets fixed values. `model.grid` lists values to tune by `inner_folds` cross validation on the training rows only. The inner scores are ROC AUC for classification, R2 for regression, and Harrell's C for survival.

### 4.4 Class balance

`balance: {method: weights}` is the default: logistic regression and random forests weight each class equally in the loss.

`balance: {method: undersample, n_bags: 10, ratio: 1.0}` trains `n_bags` models. Each one uses all samples of the smallest class and a random subset of each larger class, `ratio` times its size, drawn from the training rows only. The models' probabilities are averaged.

Neither option copies samples. Copying the few minority samples adds no information, and copies that land on both sides of a split inflate every score.

### 4.5 Evaluation

* `type: repeated_cv`, with `n_splits` and `n_repeats`. Splits are stratified by class, by event for survival, or by quintile of the target for regression, and grouped when `group_column` is set. Each sample is predicted exactly once per repeat.
* `type: leave_one_group_out`, with `column` (for example `dataset`): train on every study but one, test on the one left out, once for each study. This is the most direct test of whether a model transfers between studies. For classification, every held out study should contain both classes; otherwise its AUC is undefined, and only the recall of the class present is reported (`per_held_out_group` in `summary.json`).
* `type: holdout`, with `test_fraction`. This is one stratified, grouped split. It is useful when fitting is slow, but a single split is a noisy estimate.
* `bootstrap`: the number of resamples for intervals. Resampling is stratified by class for classification.
* `permutations`: label permutation test. The target is shuffled, and the pipeline is refitted on the same splits of repeat 0 with the hyperparameters chosen on the full data. p = (1 + number of permuted scores at or above the observed score) / (1 + permutations). Survival time and event are shuffled together. With 100 permutations the smallest possible p is about 0.0099.

### 4.6 Baseline and checks

* `compare_with_covariates_only: true` (with covariates given) scores a covariates only model on the same splits: a ridge penalised Cox model for survival, a class weighted logistic model for classification, or ridge regression. It reports the per fold difference with the corrected interval and p value, or a paired bootstrap for holdout.
* `checks.batch_column` writes a class by batch table and warns when a class appears in only one batch.

### 4.7 Output files

| file | content |
|---|---|
| `summary.json` | everything below in one place, plus row counts and exclusions |
| `folds.csv` | score of every outer fold and the hyperparameters chosen in it |
| `per_repeat.csv` | metrics on the pooled out of fold predictions of each repeat |
| `oof_predictions.csv` | every out of fold prediction (probabilities, risk scores with out of fold risk groups, or values) |
| `feature_stability.csv` | fraction of fits in which each feature was among the 50 largest coefficients or importances |
| `coefficients.csv` | final model coefficients or importances, sorted by size |
| `confounding.csv` | class by batch table |
| `roc.png`, `confusion_repeat0.png` | classification |
| `km_risk_groups.png` | survival: Kaplan Meier curves of out of fold risk tertiles, with the cut points taken from each training fold |
| `observed_vs_predicted.png` | regression |
| `model.joblib` | the final model, fitted on all rows used |
| `config_used.yaml` | the exact configuration |

### 4.8 Reading `summary.json`

* `evaluation.fold_score_corrected_interval` is the headline: the mean fold AUC, C or R2, with a 95% interval and a one sided p value for being better than chance (0.5 for AUC and C, 0 for R2). Repeated cross validation reuses samples across folds, so the plain standard error is too small. The interval uses the Nadeau and Bengio (2003) correction, (1/J + n_test/n_train) times the variance, where J is the total number of folds.
* `evaluation.per_repeat_summary` gives the mean and 2.5 to 97.5 percentiles across repeats. This spread shows how much the result depends on the particular split.
* `bootstrap_*` is a bootstrap interval on the out of fold predictions of repeat 0 for classification and regression, or on the test set for holdout.
* For classification, read the threshold metrics (balanced accuracy, recall of each class) next to AUC. `recall_exact_ci_repeat0` gives an exact (Clopper and Pearson) interval for the recall of each class, which shows how little a small class can establish: 10 of 10 correct still only bounds recall above 0.69. A high AUC can coexist with poor recall of a small class at the default decision rule.
* `covariates_only_baseline` answers whether the genes add anything. Report the interval of the difference, not only the two scores.
* `permutation_test.permuted_mean` is the score you would get by chance with this pipeline and these sample sizes.
* `final_model.note`: the evaluation estimates how well the procedure performs. The saved model is that procedure fitted once on all rows.

## 5. Select genes across datasets

`exprkit select-genes select.yaml` builds a gene list from one or more prepared datasets. Give several folders under `data`, or one folder and `split_by: <column>`. The steps:

1. **Occurrence.** A gene counts as present in a dataset when at least half the samples have a value and no single value covers `max_mode_fraction` of samples. Genes present in fewer than `min_datasets` datasets are dropped; the default is that a gene must be present in all of them.
2. **Importance in each dataset.** A signed z statistic is computed per dataset: Welch t for two classes, ANOVA F for more, Cox score test for survival, Fisher z of the correlation for regression. Datasets with a single class, or with fewer than 5 events, are skipped, because no association can be measured there.
3. **Agreement.** The per dataset z values are combined (Stouffer: their sum divided by the square root of the number of datasets). Genes whose direction disagrees between datasets are dropped (`min_sign_agreement`).
4. **Stability** (optional, `stability.n_boot`). An L1 model (lasso logistic, lasso Cox or lasso) is refitted on random halves of the pooled data, with each dataset standardised per gene first. The fraction of fits in which a gene is kept is its stability ([Meinshausen and Bühlmann 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x)).
5. **Correlation pruning.** Genes are ranked by stability, then by the size of the combined z. Walking down that ranking, a gene is skipped when its correlation with a gene already kept exceeds `max_correlation`. The process stops at `n_genes`.

Outputs are written to `output/`:
* `genes.txt`: the selected genes, ready for `genes.gene_list`.
* `gene_selection.csv`: every gene with its per dataset z, combined z, sign agreement, stability, rank, and whether it was selected or removed as correlated.
* `select_report.json`: counts for each step.

The list is chosen using these samples. See [gene_lists/README.md](../gene_lists/README.md) for how to evaluate it honestly.

## 6. Predict and evaluate

```bash
exprkit predict --model M --data data/processed/new --out preds.csv
exprkit predict --model M --matrix new_counts.csv --transform log2cpm --samples covariates.csv --out preds.csv
exprkit evaluate --model M --data data/processed/new_labelled --out results/external
```

* Genes are matched by exact symbol. The missing fraction is computed on the genes the fitted model actually uses. If it exceeds `--max-missing` (default 0.2), the run stops. Otherwise missing genes are filled with the training median, and the count is reported in `<out>_input_report.json`.
* Covariates required by the model must be present. `--allow-missing-covariates` fills them with training medians instead, and the report says so.
* The scale check compares the gene medians of the new data with the training medians. It warns when the typical shift exceeds 1 log2 unit, or when the Spearman correlation of medians is below 0.7. Either suggests a different transform or platform.
* Classification outputs a probability for each class and the most probable class. Survival outputs the risk score (log hazard, higher means worse) and the low, middle or high group, using cut points from the training data. Regression outputs the predicted value.
* `evaluate` needs the target columns in `samples.csv`, with the same names used in training. It reports metrics with bootstrap intervals; this is external validation when the new cohort is truly independent.

Saved models are pickles. Only load model files you made or got from someone you trust.

## 7. Good practice checklist

1. Run `inspect` and set `expected` counts in every prepare config.
2. Read `report.json` after prepare: unmatched IDs, unlinked samples, scale warnings.
3. Keep groups (patients, replicates) together with `group_column`.
4. Check confounding before pooling studies, using `checks.batch_column` and `batch-check`. Pool only studies that contain both classes, and test transfer with `leave_one_group_out`.
5. Never copy minority samples; use class weights or `balance: undersample`. Run prepare with `duplicates: warn` to catch copies already present in a table.
6. Choose genes inside the folds (`genes.select`), or evaluate a precomputed list on independent data.
7. Report the corrected interval, the permutation level, and the baseline difference, not only the mean score.
8. Validate on an independent cohort with `evaluate` before claiming that a model generalises.
