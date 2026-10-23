# exprkit

**Gene expression machine learning that tells you how much to trust the result.**

*Part of the [breast cancer gene expression project](../README.md). This folder is the toolkit; the project README summarises the findings.*

exprkit is a command line toolkit that takes gene expression data from raw files to a validated model you can apply to new samples. It covers classification, survival and regression. Every step reports what it dropped or changed, every score comes with an interval and a chance level, and the common traps of expression studies are checked for you: leakage, copied samples, confounded studies, and over-selected genes.

```
exprkit inspect -> exprkit prepare -> [exprkit select-genes] -> exprkit train -> exprkit predict / evaluate
look at files      clean dataset(s)     gene list + evidence      estimate with      apply, or validate on
                   one or many studies                            intervals + model  an independent cohort
```

> Research and teaching software. It is not a medical device, and its outputs must not be used for clinical decisions.

## Contents

- [What it does](#what-it-does)
- [What it protects you from](#what-it-protects-you-from)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Worked example: breast cancer](#worked-example-breast-cancer)
- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Limitations](#limitations)
- [Citing, licence and author](#citing-licence-and-author)

## What it does

| Command | Purpose |
|---|---|
| `exprkit inspect FILE` | Show sample IDs, every metadata field with its values, and the value scale of a GEO series matrix, expression table or annotation file |
| `exprkit prepare CONFIG` | Build a clean dataset from raw files: map probes to genes, log transform, link metadata, derive labels and survival columns with count checks, combine studies, detect copied profiles, normalise across platforms |
| `exprkit select-genes CONFIG` | Build a gene list across datasets: keep genes that occur in enough datasets, point the same way in each, are stable under resampling, and are not highly correlated with a better gene. The output is a list plus an evidence table |
| `exprkit train CONFIG` | Evaluate a model with nested, repeated, grouped cross validation, a held out split, or leave one study out; then fit it on all rows and save it |
| `exprkit predict` | Apply a saved model to new samples, checking genes, covariates and value scale |
| `exprkit evaluate` | Score a saved model on new labelled samples (external validation) |
| `exprkit combine`, `exprkit batch-check` | Stack prepared datasets on exact gene symbols; measure how easily studies can be told apart |
| `exprkit template prepare\|train\|select-genes` | Print a commented config with every option |

| Task | Models | Main score |
|---|---|---|
| Classification (two or more classes) | logistic, elastic net logistic, random forest, MLP | ROC AUC, plus recall of every class with exact intervals |
| Survival (time to event, censored) | Cox, elastic net Cox, DeepSurv style network | Harrell's C, plus Kaplan Meier curves of out of fold risk groups |
| Regression | ridge, elastic net, random forest | R², MAE, correlation |

**Supported inputs:**
* GEO series matrices with their platform annotation tables.
* Any CSV or TSV expression table, with genes in rows or in columns.
* Your own sample sheets.
* Ready-made tables that hold labels or survival data in the same file.

## What it protects you from

| Trap | What exprkit does |
|---|---|
| Information from test samples leaking into training (filtering, scaling, gene selection or tuning on all samples) | Every step is fitted inside each training fold, and tuning is nested ([Ambroise and McLachlan 2002](https://pmc.ncbi.nlm.nih.gov/articles/PMC124442)) |
| Labels assigned by position in a file | Labels come from metadata rules that must match each sample exactly once, checked against known counts |
| Balancing classes by copying rows | Class weights or an undersampling ensemble inside training folds; `prepare` detects copied profiles already in a table |
| Related samples split across train and test | Patient and replicate groups stay in one fold |
| A class that occurs in only one study | A warning, a batch separability check, and leave one study out evaluation ([Nygaard, Rødland and Hovig 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4679072/)) |
| A single number without uncertainty | Corrected intervals for repeated CV ([Nadeau and Bengio 2003](https://doi.org/10.1023/A:1024068626366)), stratified bootstrap, exact intervals for small classes, and a permutation chance level ([Ojala and Garriga 2010](https://www.jmlr.org/papers/v11/ojala10a.html)) |
| Genes that add nothing beyond clinical factors | A covariates only model is scored on the same folds, and the difference is reported with its interval |
| Silent mismatches on new data | `predict` stops when too many model genes are missing, and warns when the scale or platform differs |

[docs/pitfalls.md](docs/pitfalls.md) explains each trap with a worked example and the test that covers it.

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/AdebayoAAdeniyi/breast-cancer-gene-expression.git
cd breast-cancer-gene-expression/exprkit
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # add ,deep for the DeepSurv model: pip install -e ".[dev,deep]"
exprkit --help
```

## Quick start

```bash
# 1. Look at your files
exprkit inspect data/raw/GSExxxxx_series_matrix.txt.gz

# 2. Describe the data and build a clean dataset
exprkit template prepare > my_prepare.yaml          # edit: files, label field, ID mapping, transform
exprkit prepare my_prepare.yaml

# 3. Optional: build a gene list across studies
exprkit template select-genes > my_select.yaml
exprkit select-genes my_select.yaml

# 4. Train and evaluate
exprkit template train > my_train.yaml              # edit: task, target, genes, model, evaluation
exprkit train my_train.yaml

# 5. Use the saved model
exprkit predict  --model results/my_model/model.joblib --data data/processed/new_cohort --out predictions.csv
exprkit evaluate --model results/my_model/model.joblib --data data/processed/labelled_cohort --out results/external
```

A minimal prepare config for a GEO series:

```yaml
output: data/processed/my_study
sources:
  - name: GSE22820
    expression: {path: data/raw/GSE22820_series_matrix.txt, format: geo_series_matrix}
    transform: log2
    id_mapping: {path: data/raw/GPL6480-9577.txt, id_column: ID, symbol_column: GENE_SYMBOL}
    columns:
      label:
        type: category
        from: source_name_ch1
        rules: {normal: "(?i)normal", tumour: "(?i)cancer"}
        expected: {normal: 10, tumour: 176}      # stop if the counts differ from the GEO record
```

and a minimal train config:

```yaml
data: data/processed/my_study
output: results/my_model
task: classification
target: {column: label, positive_class: tumour}
genes: {filter: {top_k_variance: 2000}}
model: {type: logistic, grid: {C: [0.001, 0.01, 0.1]}}
evaluation: {type: repeated_cv, n_splits: 5, n_repeats: 20, permutations: 200}
```

## Worked example: breast cancer

The [breast cancer example](examples/breast_cancer) runs everything on public GEO data: GSE22820, GSE96058 (SCAN-B), GSE15852, GSE42568 and GSE70947. The [ready-made tables example](examples/user_tables) applies the same checks to tables that had already been combined and labelled. [docs/case_study_breast_cancer.md](docs/case_study_breast_cancer.md) walks through every decision and number.

| Question | Result |
|---|---|
| Tumour vs normal within one study (GSE22820, 176 vs 10) | AUC 1.00 in all 20 repeats. Normal recall 10 of 10, yet its exact 95% interval is 0.69 to 1.00 |
| Same question across four studies (218 normal, 471 tumour), each study held out in turn | AUC 0.89 to 1.00 on the unseen study, but the decision threshold does not transfer: all 43 normals of one study were called tumour |
| A pooled table where every normal came from one study (pooled AUC 0.996) | A model trained on it called 173 of 176 unseen tumours from the normals' study "normal": it had learned the study, not the tissue |
| Overall survival from clinical factors (3,273 patients, 336 deaths) | C = 0.76; age alone gives 0.73 |
| Adding 50 genes chosen inside each training fold | C = 0.79; gain +0.024 (95% interval +0.009 to +0.039) on the same folds |
| PAM50 subtype, five classes | macro AUC 0.977, accuracy 0.85 |

<p align="center">
  <img src="docs/figures/leave_one_study_out.png" width="860" alt="Leave one study out: AUC and normal recall on each held out study for three gene sets">
</p>
<p align="center">
  <img src="docs/figures/survival_risk_groups_infold.png" width="520" alt="Kaplan Meier curves of out of fold low, middle and high risk groups in GSE96058">
</p>

## Documentation

| Document | What it covers |
|---|---|
| [docs/user_guide.md](docs/user_guide.md) | Every command, config option, output file and check |
| [docs/pitfalls.md](docs/pitfalls.md) | 22 common ways expression analyses go wrong, and how each is handled |
| [docs/case_study_breast_cancer.md](docs/case_study_breast_cancer.md) | The full breast cancer analysis with interpretation |
| [examples/](examples) | Ready to run configs |
| [gene_lists/](gene_lists) | Reusable gene lists and how to use a list without fooling yourself |
| [data/README.md](data/README.md) | Where data go and what is never committed |

## Repository layout

```
exprkit/
├── src/exprkit/            the package
│   ├── cli.py              the exprkit command
│   ├── io.py               GEO series matrices, tables, annotation, prepared dataset format
│   ├── prepare.py          raw files -> prepared dataset (linking, transform, mapping, duplicates)
│   ├── samples.py          labels, survival, covariates and groups from metadata
│   ├── transform.py        log transforms, scale checks, per sample rank normalisation
│   ├── mapping.py          probe or transcript IDs -> gene symbols
│   ├── integrate.py        exact symbol alignment, combining studies
│   ├── filtering.py        fold safe gene filter
│   ├── selection.py        gene selection: occurrence, importance, agreement, stability, correlation
│   ├── balance.py          undersampling ensemble
│   ├── models.py           model pipelines and nested tuning
│   ├── evaluation.py       splits, out of fold predictions, permutation test, batch checks
│   ├── metrics.py          AUC, Harrell's C, R², bootstrap, exact and corrected intervals
│   ├── train.py            the train command
│   ├── bundle.py           saved models: predict and external evaluation
│   ├── deepsurv.py         neural Cox model (optional, needs torch)
│   ├── plots.py            figures
│   ├── simulate.py         simulated data for the tests
│   └── templates/          commented config templates
├── examples/
│   ├── breast_cancer/      GEO studies: configs, download scripts, run.sh
│   └── user_tables/        ready-made tables: configs and checks
├── gene_lists/             reusable gene lists with provenance
├── docs/                   user guide, pitfalls, case study, figures
├── tests/                  unit and end to end tests
├── scripts/                helper scripts (GENCODE promoter regions)
├── data/                   your data (not committed)
└── results/                outputs (not committed)
```

## Testing

```bash
pytest
```

The 51 tests run in about 20 seconds on simulated data with effects planted on purpose, so the right answer is known in advance. They cover the readers, labels, mapping, filtering, gene selection, class balance, rank normalisation, every model type, the metrics (checked against scikit-survival and slow reference implementations), and the full command line workflow. GitHub Actions runs them on every push (`../.github/workflows/tests.yml`).

## Limitations

* **Log scale input.** Values must be on a log scale before modelling. `prepare` offers log2, log2(x+1) and log2 CPM, but no TMM or DESeq2 normalisation.
* **Cross-platform models.** Rank normalisation removes scale differences between platforms, not platform specific gene behaviour. A model moved to a new platform or cohort usually needs its decision threshold recalibrated on labelled samples from that cohort.
* **Gene symbols.** Symbols are matched exactly after upper casing; renamed genes are not reconciled.
* **Coefficients are not biology.** Coefficients of penalised models on correlated genes are not stable measures of biological importance. Read `feature_stability.csv` and treat gene lists as hypotheses.

## Citing, licence and author

If you use exprkit, please cite it using [CITATION.cff](../CITATION.cff); GitHub shows a "Cite this repository" button for it. The methods it relies on are cited in the documentation.

Released under the [MIT licence](LICENSE). The licence covers the code only; public datasets keep their own terms of use and are not redistributed.

Adebayo A. Adeniyi. The breast cancer analysis began as a project for CMPUT 606 (Bioinformatics), University of Alberta, Winter Term 2025. Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).
