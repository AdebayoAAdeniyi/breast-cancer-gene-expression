# Case study: breast cancer expression, from GEO files to honest numbers

This case study runs exprkit on two public breast cancer studies. It shows each decision, the result, and what the result does and does not support. Every number below comes from the files in `results/`, produced on 22 October 2026 by the configs in `examples/breast_cancer/` and `examples/user_tables/`.

## 1. Questions

1. Can bulk gene expression separate primary breast tumours from normal breast tissue, and how certain can we be with only 10 normal samples?
2. For overall survival, how well do standard clinical factors predict risk, and does gene expression add to them?

The first question is easy in biological terms. Tumour and normal breast tissue differ strongly in cell composition (normal breast is rich in fat and stroma), so its value here is as an exercise in evaluating a small class honestly. The second question is harder, and it is the one that matters clinically.

## 2. Data, and what inspecting it revealed

| | GSE22820 | GSE96058 (SCAN-B) |
|---|---|---|
| Technology | Agilent 4x44K microarray (GPL6480) | RNA sequencing, HiSeq 2000 (3,069 profiles) and NextSeq 500 (340 profiles) |
| Samples | 176 primary breast cancers, 10 normal breast tissue | 3,409 profiles = 3,273 tumours + 136 technical replicates |
| Label source | `source_name_ch1`: "primary breast cancer" or "normal breast tissue" | all tumours |
| Values as distributed | positive ratios, 0.01 to 20,566, median 0.85 | log2(FPKM + 0.1) for 30,865 gene symbols (series processing notes) |
| Outcome | none | overall survival: 336 deaths among 3,273 patients; follow up up to 2,474 days, median observed time 1,590 days |

Running `exprkit inspect` before writing any config changed three decisions:

* **The GSE22820 values are not logged.** They were normalised in GeneSpring 7.3.1 with "default normalization procedures" (series notes), and are centred near 1, which is consistent with ratios to each gene's median. They therefore need `transform: log2`. Fitting a linear model on unlogged values would have let a handful of very large values dominate.
* **Labels come from metadata.** In this file the 10 normal samples happen to be the last 10 columns, so labelling by position would have worked here by luck. The metadata rule, checked against the 10 and 176 on the GEO record, works whatever the order.
* **Replicates are named `F<n>repl`.** The pattern `repl$` groups each replicate with its tumour, and prepare confirms that exactly 136 patients have two profiles. Survival models keep one profile per patient.

GSE118389, considered as a third cohort, turned out to be single cell RNA sequencing of 1,534 cells from 6 triple negative tumours. Single cells are not comparable to bulk tumour profiles, so it is not used.

## 3. Preparing GSE22820

| Step | Count |
|---|---|
| Probes in the series matrix | 41,000 |
| Probes without a gene symbol (controls and unannotated features), dropped | 10,064 |
| Probes annotated to several genes | 0 |
| Genes after averaging probes of the same gene | 19,595 (8,076 genes had more than one probe) |
| Samples linked to metadata | 186 of 186 |

## 4. Tumour versus normal (GSE22820)

**Design.**
* **Model.** Logistic regression with an L2 penalty and balanced class weights, on the 2,000 most variable genes.
* **Fitting inside folds.** The gene filter, scaling and choice of penalty (C in {0.001, 0.01, 0.1, 1}, by 3 fold inner cross validation) are all fitted inside each training fold.
* **Evaluation.** Stratified 5 fold cross validation repeated 20 times (100 outer fits), so each normal sample is tested once per repeat.
* **Chance level.** A permutation test with 200 label shuffles.
* **Why one study only.** All normal samples come from this one study, so the model is trained and tested within it.

**Results** (`results/gse22820_cancer_vs_normal/summary.json`)

| Quantity | Value |
|---|---|
| ROC AUC, every one of the 20 repeats | 1.000 |
| Balanced accuracy, mean across repeats (2.5 to 97.5 percentile) | 0.997 (0.994 to 1.000) |
| Normal recall at the default threshold, repeat 0 | 10 of 10; exact 95% interval 0.69 to 1.00 |
| Cancer recall at the default threshold, repeat 0 | 175 of 176; exact 95% interval 0.97 to 1.00 |
| Out of fold probability of cancer: highest normal sample, lowest tumour sample | 0.15 and 0.44 |
| Permutation test: mean AUC with shuffled labels, 95th percentile | 0.49, 0.71 |
| Permutation p value | 0.005 (the smallest possible with 200 shuffles) |
| Penalty chosen | C = 0.001 in all 100 folds |

**How to read these numbers.**

* **Perfect separation, but limited certainty.** Every repeat separated tumours from normals perfectly, yet ten normal samples cannot establish high certainty. The exact interval for normal recall still runs from 0.69 to 1.00. The corrected cross validation interval collapses to 1.000 to 1.000 only because every fold scored the same, which is a limitation of fold based intervals when performance hits the ceiling, not evidence of zero uncertainty.
* **Chance can look good with a tiny class.** With shuffled labels, 5% of runs reached an AUC of 0.71 or more. With a class of 10, chance alone can produce numbers that look respectable, which is why the permutation level is part of the report.
* **The chosen penalty reflects ties.** C = 0.001 was chosen every time because all penalties tied at a perfect inner AUC, and the search keeps the first of tied values.
* **Recognisable stroma and fat genes.** The most stable genes across fits (`feature_stability.csv`) include COL10A1, COL11A1, MMP11 and COMP, which are higher in tumours (mean log2 differences of 4.7 to 6.4), and LEP, RBP4, CA4 and ANGPTL7, which are higher in normal tissue (4.6 to 5.5). This pattern is consistent with tumour stroma versus fatty normal breast. It is a description of what drives this classifier, not a discovery.
* **A possible processing batch.** The normal samples carry a different GEO "last update" date from the tumours (27 April 2011 versus 14 June 2011) and sit together at the end of the file. If they were also hybridised separately, part of the separation could be a processing batch. The data available cannot test this; only an independent normal cohort could.

## 5. Overall survival in GSE96058: the clinical bar

**Design.**
* **Cohort.** 3,273 patients (replicate profiles dropped), with 336 deaths.
* **Model.** A Cox model with a small ridge penalty.
* **Evaluation.** Stratified (by event) and grouped (by patient) 5 fold cross validation, repeated 10 times, plus 50 permutations.
* **Missing values.** Missing clinical values among the 3,273 patients (tumour size 34, node status 93, grade 61, ER 200, HER2 122; GEO writes them as "NA") are imputed inside each training fold.

| Model | Mean fold C (95% corrected interval) |
|---|---|
| Age only | 0.726 (0.694 to 0.759) |
| Age + tumour size | 0.749 (0.721 to 0.778) |
| Age + size + node status + grade + ER + HER2 | 0.763 (0.736 to 0.789) |
| Shuffled outcomes (permutation mean) | about 0.50 |

The full clinical model is better than age and size on the same folds: the difference is +0.013, with a corrected 95% interval from 0.000 to 0.027 (p = 0.048).

**How to read this.** Age alone reaches C = 0.73. Overall survival counts deaths from any cause, and in a population based cohort many deaths in older patients are not from breast cancer. So a gene model reaching C = 0.75 for overall survival would be no better than age and tumour size. Any claim that expression is prognostic has to be made against the 0.76 of the clinical model, on the same folds, as section 6 does. Breast cancer specific or recurrence free survival would be a cleaner target, but GSE96058 provides overall survival only.

## 6. Does expression add to the clinical factors?

**Data.** The GSE96058 expression file has 30,865 gene symbols as log2(FPKM + 0.1); after upper casing, two symbols merge, leaving 30,863. Its columns are the sample titles, and all 3,409 profiles link to their metadata. A quarter of all values sit at the floor of log2(0.1) = −3.32, which means zero FPKM; the quasi constant filter removes genes that are almost always at the floor.

**Design** (`train_survival.yaml`).
* **Model.** Elastic net Cox (l1 ratio 0.5) on the 1,000 most variable genes, plus age, tumour size, node status, grade, ER and HER2.
* **Tuning.** The penalty is tuned by 3 fold inner cross validation. The smallest grid value (0.003) failed numerically in every fold and was skipped; alpha = 0.03 was chosen in all 10 outer folds.
* **Evaluation.** 5 fold cross validation repeated twice, on the same patients and splits as a clinical only Cox model.

| Model (same 10 folds) | Mean fold C (95% corrected interval) |
|---|---|
| Clinical only (age, size, node, grade, ER, HER2) | 0.761 |
| Genes + clinical, elastic net Cox | 0.782 (0.744 to 0.821) |
| Difference | +0.021 (+0.002 to +0.039), p = 0.03 |

Kaplan Meier curves of out of fold risk tertiles separate clearly (log rank p < 10⁻⁵⁰), with the cut points taken from each training fold.

![Out of fold risk groups, in-fold gene selection](figures/survival_risk_groups_infold.png)

*Figure: the model with 50 genes chosen inside each training fold (section 10); each patient's risk group comes from the fold in which it was a test patient.*

**How to read this.**
* **A modest gain.** Expression adds about 0.02 to the C index of a model that already has the standard clinical factors. The corrected interval excludes zero, but only just. The estimate rests on 10 folds and 336 events in a single cohort, so it should be confirmed in an independent cohort before being called prognostic.
* **What the final model kept.** It keeps 29 non zero terms. Age and tumour size have the largest coefficients. The genes selected in every fit include CA9 (higher risk), and SCUBE2, CLIC6 and PCAT18 (lower risk). SCUBE2 is one of the genes of the 21 gene Oncotype DX recurrence score, which makes its appearance plausible. Gene lists from penalised models on correlated genes are unstable, so treat them as hypotheses (`feature_stability.csv`).
* **A design choice to revisit.** The elastic net penalty applies to the clinical covariates too, and the categorical clinical factors (node status, grade, ER, HER2) were shrunk to zero once genes were available. Genes that track grade and ER status can stand in for them. A model that leaves the clinical factors unpenalised would answer the "beyond clinical" question more strictly.

## 7. Why the studies are not pooled: the batch check

A logistic model trained to tell GSE22820 tumours from GSE96058 tumours, using the 17,786 genes the studies share, reaches a cross validated AUC of 1.00. The two studies differ in platform and processing (microarray log ratios versus RNA sequencing log2 FPKM), and the difference is total. All normal samples come from GSE22820, so a tumour versus normal model trained on pooled data could score perfectly by recognising the study alone. This is why section 4 stays within one study.

## 8. PAM50 subtype (multiclass)

**Design** (`train_pam50.yaml`).
* **Data.** 3,273 tumours, one profile each: LumA 1,657, LumB 729, Basal 339, Her2 327, Normal like 221.
* **Model.** Multinomial logistic regression on the 2,000 most variable genes; C = 0.01 was chosen in the final fit.
* **Evaluation.** 5 fold cross validation repeated twice.

| Quantity | Value |
|---|---|
| Macro one versus rest ROC AUC | 0.977 (95% corrected interval 0.974 to 0.980) |
| Accuracy / balanced accuracy | 0.854 / 0.836 |
| Recall, repeat 0 (exact 95% interval) | Basal 0.91 (0.88 to 0.94), Her2 0.85 (0.80 to 0.88), LumA 0.87 (0.86 to 0.89), LumB 0.82 (0.79 to 0.85), Normal like 0.71 (0.65 to 0.77) |

**How to read this.**
* **The errors make biological sense.** Most errors are LumA called LumB (125) and LumB called LumA (98). These two subtypes lie on a continuum of proliferation, so the boundary between them is the least sharp. Normal like tumours are the hardest class (recall 0.71), and are most often called LumA. Basal tumours are the most distinct (recall 0.91).
* **Recovering a label, not a discovery.** The labels were themselves computed from these RNA sequencing data, with a nearest centroid method on the 50 PAM50 genes. A model that recovers them at 85% accuracy from a different gene set shows that the multiclass workflow works on real data; it is not an independent finding. To predict subtype on new data, training on the PAM50 gene list (`genes.gene_list`) would be the more faithful design.

## 9. Ready-made tables: what checking them revealed

Expression analyses often start from tables that were already combined, labelled and split. Four such files were run through exprkit: `train_data.csv`, `val_data.csv` and `test_data.csv` (79 genes, a 0/1 label), and `merged_expression_survival.csv` (216 genes plus survival). `examples/user_tables/` holds the configs.

**The classification tables.**

| Check | Finding |
|---|---|
| Duplicate profiles (`duplicates: drop`) | The 2,868 normal rows in the training file are 8 distinct profiles, each copied 344 to 377 times |
| Normals in validation and test | 1 each; getting 1 of 1 right bounds normal recall only above 0.03 (exact 95% interval) |
| Source of each row (`assign_study.py`, by correlation with GEO profiles) | 176 tumours + 10 normals from GSE22820, 3,409 tumours from GSE96058; the split is sharp (best correlation with GSE96058 at most 0.62 for the first group, at least 0.78 for the second) |

So every normal comes from one study, and 95% of the tumours from another. Three experiments show what that does:

| Experiment (79 genes, copies removed) | Result |
|---|---|
| Pooled, as the tables were built: 10 normals vs 3,585 tumours, 10 × 5 fold CV | AUC 0.996 (0.992 to 1.000). But by study, 99.95% of GSE96058 tumours and only **39.5%** of GSE22820 tumours are called tumour |
| Train normal (GSE22820) vs tumour (GSE96058 only); score the 176 GSE22820 tumours, never seen | **173 of 176 called normal**. The model learned which study, not which tissue. `predict` also warned of a different scale (gene medians correlate −0.49) |
| Within GSE22820 only (176 vs 10), 20 × 5 fold CV | AUC 0.968 (0.912 to 1.000); normal recall 6 of 10 (exact interval 0.26 to 0.88) |

The near perfect pooled AUC came mostly from telling the two studies apart. Within the one study that has both classes, the 79 genes do well but less well than 2,000 variable genes from the GEO files (section 4: AUC 1.000, 10 of 10 normals).

**The survival table** (4,297 rows).
* 3,409 rows are GSE96058. Every survival time and event matches GEO, and expression correlates 0.995 with the GEO values.
* 888 rows come from four other GEO series. Their events could not be checked against source. For 734 of them, tumour size equals 12.50 + 0.118 × age exactly (correlation 1.000): it had been filled in from age. `missing_when` sets those values to missing, so imputation happens inside the folds instead.

| Model (same folds as its baseline) | C | Gain over age + size |
|---|---|---|
| 216 genes + age + size, GSE96058 rows | 0.785 (0.750 to 0.820) | +0.036 (+0.012 to +0.059) |
| 216 genes + age + size, all 4,161 patients (465 deaths) | 0.757 (0.729 to 0.785) | +0.035 (+0.017 to +0.054) |

The 216 genes were themselves chosen using samples from these studies, so these gains are likely optimistic (section 10).

## 10. Choosing genes: lists made outside the folds versus inside

A gene list chosen on some samples carries information about them. With 2,000 genes of pure noise and 60 samples, choosing the best 20 on all samples and then cross validating gives an AUC of 0.94; choosing inside each fold gives 0.52 (`tests/test_selection.py`). Three ways of using genes for overall survival in GSE96058, each with the full clinical factors and compared on the same folds with the clinical model:

| Genes | C | Gain over clinical (corrected 95% interval) |
|---|---|---|
| The 79 gene list (chosen earlier, partly on these data) | 0.777 | +0.015 (−0.000 to +0.030), p = 0.06 |
| 1,000 most variable genes, elastic net (section 6) | 0.782 | +0.021 (+0.002 to +0.039), p = 0.03 |
| **50 genes chosen inside each training fold** (Cox score, correlation ≤ 0.7) | **0.786** | **+0.024 (+0.009 to +0.039), p = 0.003** |

The in-fold version is the only one whose estimate includes the choice of genes. It is also the strongest, with 25 folds.

`exprkit select-genes` turns the occurrence, importance and correlation approach into a reproducible list with all its evidence. Two lists were made:

* **Survival** (`gene_lists/breast_cancer_survival_gse96058_50.txt`). GSE96058's two sequencing platforms were treated as two datasets. 16,433 genes are present on both and point the same way on both. The 50 kept include CA9, GREB1, FLT3 and VAV3.
* **Tumour versus normal across four studies** (`gene_lists/breast_tumour_normal_4studies_50.txt`, section 11). 4,211 of 11,310 shared genes change in the same direction in all four studies. The top genes are recognisable tumour stroma and proliferation genes (COL10A1, COMP, FN1, BGN, VCAN, EZH2, RRM2), and genes higher in normal tissue (IGFBP6, LYVE1, GPC3, NDN).

  Of the 79 gene list, 64 genes are measured on all four platforms. Only 26 of those change in the same direction in all four studies, and their typical strength is lower (median |combined z| 4.4, against 7.7 for all consistent genes). Only RRM2 is shared with the new list.

## 11. Tumour versus normal across four studies

More normal samples, honestly obtained, means more studies that each contain both classes. Three GSE studies were added:

| Study | Platform | Normal | Tumour |
|---|---|---|---|
| GSE22820 | Agilent 4x44K | 10 | 176 |
| GSE15852 | Affymetrix U133A | 43 (paired) | 43 |
| GSE42568 | Affymetrix U133 Plus 2 | 17 | 104 |
| GSE70947 | Agilent | 148 (adjacent to tumour, paired) | 148 |
| **Total** | | **218** | **471** |

* **Preparation.** Each study's probes were mapped with its own platform table (GSE15852's MAS 5.0 values were log2 transformed). The studies were combined on their 11,310 shared genes and ranked within each sample (`normalise: per_sample_rank`). Tumour and normal from the same patient stay in one fold.
* **A label check that catches a real error.** GSE70947 calls its normal samples "breast tissue adjacent to tumor". A rule that looks for "tumour" in that field would label them tumour. With rules on that field, prepare stops: "148 samples match more than one rule". That study's labels come from its `tissue` field instead.

**Leave one study out** (train on three studies, test on the fourth), AUC of the held out study:

| Genes | GSE15852 | GSE22820 | GSE42568 | GSE70947 |
|---|---|---|---|---|
| 2,000 most variable, logistic | 0.936 | 1.000 | 0.969 | 0.890 |
| 100 chosen in fold + undersampling ensemble | 0.886 | 1.000 | 0.932 | 0.882 |
| The 79 gene list (64 available) | 0.862 | 0.994 | 0.937 | 0.874 |
| *Pooled 5 fold CV over all four, for contrast* | | | | *0.985 overall* |

![Leave one study out](figures/leave_one_study_out.png)

**How to read this.**
* **Ranking transfers, the threshold does not.** Held out studies reach AUC 0.87 to 1.00, so the model still ranks tumours above normals in a study it never saw. But the decision threshold does not carry over. With the plain model, all 43 GSE15852 normals were called tumour (normal recall 0.00) despite an AUC of 0.94, and 21% of GSE22820 tumours were called normal. A model moved to a new cohort needs its threshold recalibrated on some labelled samples from that cohort.
* **Pooled CV is too optimistic.** Pooled cross validation (AUC 0.985, normal recall 0.94) overstates how a model does on a new study, because every fold contains samples from every study.
* **Rank normalisation does not remove batches.** Normal samples of GSE70947 and GSE15852 can be told apart with AUC 1.00, and tumours of GSE22820 and GSE70947 too. Rank normalisation removes scale differences, not platform specific gene behaviour. This is why each study must contain both classes.
* **Balancing and selection did not help here.** Undersampling and in-fold selection of 100 genes did not improve transfer. With 218 real normals and class weights, the imbalance is no longer the limiting factor; differences between studies are.

## 12. Limitations

* Section 4 uses ten normal samples from one study. Section 11 adds 208 more from three studies, but their definitions of "normal" differ (normal breast versus tissue adjacent to a tumour).
* Survival is analysed in one cohort with internal cross validation only; there is no external validation cohort.
* Overall survival mixes cancer and non cancer deaths.
* The clinical models use simple coding: one hot categories with a ridge penalty, and linear age and size effects.

## References

* Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene expression data. PNAS 2002;99(10):6562 to 6566.
* Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Statistics in Medicine 1996;15(4):361 to 387.
* Meinshausen N, Bühlmann P. Stability selection. Journal of the Royal Statistical Society B 2010;72(4):417 to 473.
* Nadeau C, Bengio Y. Inference for the generalization error. Machine Learning 2003;52(3):239 to 281.
* Nygaard V, Rødland EA, Hovig E. Methods that remove batch effects while retaining group differences may lead to exaggerated confidence in downstream analyses. Biostatistics 2016;17(1):29 to 39.
* Ojala M, Garriga GC. Permutation tests for studying classifier performance. Journal of Machine Learning Research 2010;11:1833 to 1863.
* Paik S, Shak S, Tang G, et al. A multigene assay to predict recurrence of tamoxifen treated, node negative breast cancer. New England Journal of Medicine 2004;351(27):2817 to 2826.
* Clopper CJ, Pearson ES. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika 1934;26(4):404 to 413.
