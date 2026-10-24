# Word vs. character features for SMS spam detection

## Abstract
This exploratory study compares word unigram and character 3–5-gram representations for SMS spam classification. Both use the same Multinomial Naive Bayes classifier and stratified five-fold evaluation on the UCI SMS Spam Collection. Results are reported as pooled out-of-fold metrics and fold-to-fold variation. The comparison is descriptive and limited to this dataset.

## Research question
Does a word-based or character-based representation change spam detection performance, including error rates on shorter and longer messages?

## Dataset
The UCI SMS Spam Collection contains 5,574 labeled SMS messages. It is English-language, historical, and class-imbalanced; it should not be treated as representative of current messages or all populations.

Source: UCI Machine Learning Repository, [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection), DOI: 10.24432/C5CC84.

## Experimental design
- **Evaluation:** stratified five-fold cross-validation with random seed 42. Each message is held out for evaluation once. The same folds are used for both representations.
- **Classifier:** Multinomial Naive Bayes with Laplace smoothing (alpha=1.0); implemented with Python's standard library.
- **Representations:** word unigrams; overlapping character n-grams of lengths 3, 4, and 5.
- **Length groups:** message length is whitespace-separated word count. Within each fold, training data set separate median cutoffs for ham and spam. Held-out messages are grouped using the cutoff for their true class. This keeps the length comparisons within class and avoids using held-out text to choose the cutoff.
- **Metrics:** spam recall (fraction of spam caught), ham false-positive rate (fraction of legitimate messages wrongly flagged), and accuracy. Pooled metrics combine all out-of-fold predictions; fold mean and sample standard deviation show variation across the five folds. The standard deviation is descriptive, not a confidence interval.

Training-set class-specific median cutoffs by fold: ham 11 words and spam 25 words in all folds.

## Exact-duplicate audit
Messages were normalized by case-folding and collapsing whitespace; no fuzzy duplicate matching was used. The audit found 290 exact normalized duplicate groups, representing 415 repeated rows beyond the first copy. 241 duplicate groups (601 rows) appeared in more than one random stratified fold. Therefore, an evaluation message may have an exact normalized copy in its training fold, and the reported cross-validation scores may be optimistic for unseen messages. This version does not include duplicate-grouped cross-validation; keeping duplicate groups together is an important next evaluation step.

## Results: overall

| Representation | Out-of-fold n | Pooled accuracy | Accuracy mean ± SD | Pooled spam recall | Spam recall mean ± SD | Pooled ham false-positive rate | Ham false-positive mean ± SD | Missed spam | False spam flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Word unigrams | 5574 | 0.981 | 0.981 ± 0.0019 | 0.949 | 0.949 ± 0.0198 | 0.014 | 0.014 ± 0.0038 | 38 | 69 |
| Character 3-5 grams | 5574 | 0.984 | 0.984 ± 0.0005 | 0.950 | 0.950 ± 0.0138 | 0.011 | 0.011 ± 0.0024 | 37 | 51 |

## Results: length subgroups

### Spam recall
| Model | Group | n | Pooled recall | Fold mean ± SD | Missed spam |
|---|---|---:|---:|---:|---:|
| Word unigrams | spam short | 396 | 0.962 | 0.963 ± 0.0177 | 15 |
| Word unigrams | spam long | 351 | 0.934 | 0.933 ± 0.0265 | 23 |
| Character 3-5 grams | spam short | 396 | 0.960 | 0.960 ± 0.0200 | 16 |
| Character 3-5 grams | spam long | 351 | 0.940 | 0.940 ± 0.0244 | 21 |

### False-positive rate among legitimate messages
| Model | Group | n | Pooled false-positive rate | Fold mean ± SD | False spam flags |
|---|---|---:|---:|---:|---:|
| Word unigrams | ham short | 2605 | 0.020 | 0.020 ± 0.0058 | 51 |
| Word unigrams | ham long | 2222 | 0.008 | 0.008 ± 0.0035 | 18 |
| Character 3-5 grams | ham short | 2605 | 0.015 | 0.015 ± 0.0036 | 39 |
| Character 3-5 grams | ham long | 2222 | 0.005 | 0.005 ± 0.0020 | 12 |

## Interpretation
Compared with the word model, the character model's pooled spam recall changed by **+0.13 percentage points** and its pooled ham false-positive rate changed by **-0.37 percentage points**. Positive recall change means more spam was caught; positive false-positive change means more legitimate messages were wrongly flagged. Consider pooled results together with fold variation and the different costs of these two error types. The experiment does not establish that either representation is generally superior.

## Limitations
The study uses one old, English-only dataset. Cross-validation measures variation across folds from this dataset; it does not test transfer to newer messages or different populations. Fold standard deviations are not formal confidence intervals. Naive Bayes is a simple baseline, and neither representation understands message meaning or context. A follow-up could evaluate a newer dataset and pre-register a broader set of metrics.

## Reproducibility
Run `python main.py` from this folder. The program writes pooled summary metrics to `results/model_comparison_metrics.csv`, length-group results to `results/length_group_metrics.csv`, per-fold metrics to `results/fold_metrics.csv`, all held-out predictions to `results/out_of_fold_predictions.csv`, and errors to `results/misclassified_messages.csv`.
