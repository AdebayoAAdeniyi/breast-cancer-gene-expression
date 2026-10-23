# Common pitfalls in gene expression machine learning

Expression datasets have thousands of genes, often fewer than a few hundred samples, small classes, censored outcomes, and data pooled from different studies. Under those conditions it is easy to get a result that looks excellent and does not hold up on new data. This page describes the pitfalls exprkit is built to avoid. For each one, it explains why the pitfall matters, what exprkit does about it, and which test in `tests/` demonstrates that.

## Data preparation

### 1. Labels assigned by position
**Pitfall.** You assume that, say, "the last 10 columns are the normal samples". Column order in a downloaded file is not guaranteed, so one wrong assumption mislabels samples silently.
**exprkit.** Labels come from a metadata field, matched with regular expressions. Every sample must match exactly one rule, and `expected` counts tie the rules to numbers you know independently, such as the sample counts on the GEO record.
**Test.** `test_labels_from_metadata_not_position`, `test_labels_fail_loudly`

### 2. Approximate gene name matching
**Pitfall.** Fuzzy string matching is used to "rescue" genes that are missing in one dataset. Gene families differ by one character: `difflib` similarity is 0.75 for ESR1 and ESR2, and 0.80 for HOXA1 and HOXA2. At a 0.7 cut off, one gene is silently replaced by a different one.
**exprkit.** Matching is by exact symbol only. Missing genes are reported, never substituted.
**Test.** `test_fuzzy_matching_would_swap_real_genes`, `test_align_exact_and_nan`

### 3. Filling missing genes with zero
**Pitfall.** Zero is a real value. On a log scale it means a specific expression level, so filling with zeros invents measurements.
**exprkit.** Missing values stay missing until model fitting. They are then imputed with the median of the training rows only, and the count is reported.
**Test.** `test_align_exact_and_nan`

### 4. Wrong value scale
**Pitfall.** Linear models and standardisation assume roughly symmetric, log scale values. Raw intensities, ratios or counts are dominated by a few very large values. The opposite mistake, logging data that are already logged, compresses everything.
**exprkit.** `inspect` and `prepare` report the value range. Prepare warns when the chosen transform looks wrong, and `predict` compares the scale of new data with the training data.
**Example.** GSE22820 is distributed as positive ratios from 0.01 to 20,566 with a median of 0.85, so it needs `transform: log2` (see the breast cancer case study).
**Test.** `test_transforms_and_scale_warnings`, `test_predict_from_matrix_with_scale_warning`

### 5. Text that means "missing"
**Pitfall.** GEO metadata often writes `NA`. Read naively, `NA` becomes a category, so a model learns a class called "NA", or a number parser silently drops it.
**exprkit.** `NA`, `N/A`, `unknown`, empty strings and similar values are missing for every column type.
**Test.** `test_derive_columns_types`, `test_na_text_is_missing`

### 6. Losing decimals
**Pitfall.** Parsing numbers with the regular expression `\d+` turns 22.5 into 22.
**exprkit.** Number parsing keeps decimals and scientific notation.
**Test.** `test_number_and_event_parsing`

## Evaluation

### 7. Information leaking from test samples
**Pitfall.** Genes are filtered or selected, data are scaled, or hyperparameters are tuned on all samples before cross validation. The test folds then influence the model, and the estimate is optimistic. With labels involved, the bias can be dramatic ([Ambroise and McLachlan 2002](https://pmc.ncbi.nlm.nih.gov/articles/PMC124442)).
**exprkit.** Every preprocessing step is part of the model pipeline and is refitted on each training fold. Tuning happens in an inner cross validation, which makes the whole evaluation a nested cross validation.
**Example.** With 2,000 genes of pure noise and 60 samples, choosing the 20 best genes on all samples and then cross validating gives AUC 0.94; choosing them inside each fold gives 0.52.
**Test.** `test_filter_fits_on_training_rows_only_and_ignores_nan_variance`, `test_selection_outside_cv_is_optimistic`

### 8. A single small test set
**Pitfall.** With 10 normal samples, an 80/10/10 split leaves one normal in the test set, so the "accuracy" for that class is either 0% or 100%.
**exprkit.** Repeated stratified k fold cross validation predicts every sample exactly once per repeat. The spread across repeats shows how much the result depends on the split.
**Test.** `test_repeated_cv_tests_every_minority_sample_each_repeat`

### 9. Copying minority samples to balance classes
**Pitfall.** Oversampling the same 8 normal profiles thousands of times teaches the model those 8 profiles, not the class. Done before splitting, it also puts copies of a sample on both sides of the split.
**exprkit.** Imbalance is handled with class weights in the loss, or with `balance: undersample`, an ensemble of models each trained on all minority samples and an equal random subset of the majority, drawn inside the training folds. Threshold metrics, such as the recall of each class and balanced accuracy, are reported next to AUC, with exact intervals for small classes.
**Test.** `test_undersample_ensemble_balances_without_copies`

### 10. Related samples on both sides of a split
**Pitfall.** Technical replicates or several samples from one patient are split apart. The model is then tested on a near copy of a training sample.
**exprkit.** `group` columns and `group_column` keep groups together. `one_per_group` keeps one profile per patient, and the replicate count is checked during prepare.
**Test.** `test_grouped_splits_never_share_groups`, `test_patient_groups`

### 11. A single number without uncertainty
**Pitfall.** "AUC = 0.97" says nothing about how much it would change with other samples, or about what chance would give.
**exprkit.** Reports include a corrected interval for repeated cross validation ([Nadeau and Bengio 2003](https://mlanthology.org/mlj/2003/nadeau2003mlj-inference)), stratified bootstrap intervals, and a label permutation test that gives the chance level of this exact pipeline ([Ojala and Garriga 2010](https://www.jmlr.org/papers/v11/ojala10a.html)).
**Test.** `test_paired_bootstrap_and_corrected_t`, `test_no_signal_gives_chance_auc`

### 12. No comparison with what is already known
**Pitfall.** A gene model reaches a C index of 0.75 and is called prognostic. But age and tumour size alone may reach the same value.
**exprkit.** When covariates are given, a covariates only model is scored on the same folds, and the difference is reported with its interval.
**Example.** In GSE96058, age and tumour size alone give C = 0.75 for overall survival (see the case study).

### 13. Class confounded with study
**Pitfall.** All normal samples come from one study and the tumours from several. A classifier can then reach a high score by recognising the study, not the tissue. Batch correction cannot help, because when a class occurs in only one batch, the class effect and the batch effect cannot be separated ([Nygaard, Rødland and Hovig 2016](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4679072/)).
**exprkit.** `checks.batch_column` warns when a class occurs in only one batch. `batch-check` measures how easily studies can be told apart within a single class.
**Test.** `test_batch_check_and_confounding`, `test_combine_and_batch_check`

### 14. Trusting pooled cross validation across studies
**Pitfall.** Several studies are pooled and cross validated together. Every test fold then contains samples from every study, so the model is never tested on a study it has not seen.
**exprkit.** `evaluation.type: leave_one_group_out` with `column: dataset` trains on all studies but one and tests on the one left out.
**Example.** In the breast cancer case study, pooled CV over four studies gives AUC 0.985. Leaving each study out gives 0.89 to 1.00, and in one study all 43 normals were called tumour, because the decision threshold does not transfer.
**Test.** `test_leave_one_study_out_with_selection_and_balance`

### 15. Copies hiding in a ready-made table
**Pitfall.** A table was balanced earlier by copying minority rows, and the copies have new IDs, or none.
**exprkit.** `prepare` hashes every expression profile and reports identical ones. `duplicates: drop` keeps one copy of each, and `duplicates: group` keeps them together in one fold.
**Example.** A training table with 2,868 normal rows turned out to hold 8 distinct profiles.
**Test.** `test_duplicate_profiles_found`, `test_prepare_tables_drops_copies`

## Survival analysis

### 16. Treating censored times as event times
**Pitfall.** Regressing survival time with mean squared error treats a patient who was alive at last follow up as if they died then. Censored times are lower bounds, so this biases predictions towards short survival.
**exprkit.** Survival models use the Cox partial likelihood, which treats censoring correctly. Survival time is never a regression target.

### 17. Filling in missing events
**Pitfall.** Missing event indicators are assigned at random, or predicted by a heuristic. The outcome being modelled is then partly invented.
**exprkit.** Samples with a missing or invalid event are excluded and counted, never imputed.
**Test.** `test_survival_with_covariate_baseline`

### 18. Oversampling events
**Pitfall.** The Cox likelihood compares each event with everyone still at risk. Duplicated rows change those risk sets and the meaning of the estimated hazard ratios.
**exprkit.** No resampling of rows when fitting survival models.

### 19. A C index that skips censored patients
**Pitfall.** Computing concordance only over pairs where both patients had an event discards most comparable pairs when censoring is common, and gives a different number.
**exprkit.** Harrell's C over all comparable pairs: the earlier time must be an event, and the later one may be censored ([Harrell, Lee and Mark 1996](https://doi.org/10.1002/(SICI)1097-0258(19960229)15:4%3C361::AID-SIM168%3E3.0.CO;2-4)). It is checked against scikit-survival.
**Test.** `test_harrell_c_matches_scikit_survival`, `test_events_only_c_index_differs`

### 20. Sign conventions in neural Cox models
**Pitfall.** Ad hoc rules such as "negate the scores, and flip them back if the C index looks low" fire exactly when a model is good, because negating a good ordering makes it look bad.
**exprkit.** One convention everywhere: the score is a log hazard, and higher means worse. The neural Cox loss is tested against a slow reference implementation, including tied times.
**Test.** `test_torch_loss_matches_reference_with_ties`, `test_loss_prefers_true_direction`, `test_sign_flip_heuristic_fires_on_good_models`

### 21. Risk groups defined on the test data
**Pitfall.** Splitting test patients into low, middle and high risk at the test set's own tertiles makes the groups look balanced by construction, and the cut points could never be used on a new patient.
**exprkit.** Cut points come from training data: each training fold during evaluation, and the full training set for a saved model.
**Test.** `test_risk_groups_follow_cutpoints`

## Applying a model

### 22. Silent gene mismatch on new data
**Pitfall.** New data lack some of the model's genes, or use another platform or scale. A prediction is still produced, but it means little.
**exprkit.** `predict` stops when more than a set fraction of the genes the model uses are missing. It reports which genes were filled with training medians, and warns when gene medians differ from training.
**Test.** `test_classification_train_predict_evaluate`, `test_predict_from_matrix_with_scale_warning`
