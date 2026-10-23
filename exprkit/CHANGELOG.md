# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org).

## 0.4.0 (22 October 2026)

### Added
- `exprkit select-genes`: gene lists across datasets, based on occurrence, signed importance per dataset with Stouffer combination, sign agreement, stability selection and correlation pruning. Writes `genes.txt` and a full evidence table.
- `genes.select` in train configs: univariate selection with correlation pruning inside every training fold.
- `balance: {method: undersample}`: undersampling ensemble inside training folds.
- `evaluation.type: leave_one_group_out`, for example leave one study out, with per study metrics.
- `normalise: per_sample_rank` in prepare and `exprkit combine --normalise`; saved models apply the same step to new data.
- Ready-made tables: `metadata_columns`, `index_column: null`, `map` for coded labels, `from: _id`, `missing_when`.
- `duplicates: warn | drop | group`: detection of identical expression profiles under different IDs.
- Exact (Clopper and Pearson) intervals for the recall of every class.
- Breast cancer examples across four GEO studies, the ready-made tables example, and reusable gene lists.

### Changed
- Gene columns are selected by position inside pipelines, which makes fits about 10 times faster with 20,000 or more genes.
- Risk group curves use an ordered single hue ramp.

## 0.3.0 (22 October 2026)

### Added
- The commands `inspect`, `prepare`, `train`, `predict`, `evaluate`, `combine`, `batch-check` and `template`.
- Classification, survival and regression with nested, repeated and grouped cross validation, corrected intervals, bootstrap intervals, permutation tests, and a covariates only baseline.
- Clinical only datasets, NA aware metadata parsing, and scale checks on prediction.
- The breast cancer case study on GSE22820 and GSE96058.
