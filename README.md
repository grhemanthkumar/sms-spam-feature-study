# Word vs. Character Features for SMS Spam Detection

A reproducible, exploratory machine-learning comparison of word and character n-gram features for SMS spam classification.

> **Interpretation warning:** This is a small study on one historical dataset, not a production spam filter or a benchmark of current messages. A post-hoc exact-duplicate audit found duplicates crossing the evaluation folds, so the reported cross-validation scores may be optimistic for previously unseen messages. See the [report](report.md) for the counts and details.

## Research question

Does changing the text representation from word unigrams to character 3–5-grams change SMS spam classification performance, including error patterns across short and long messages?

## Dataset

The UCI SMS Spam Collection contains 5,574 labeled English messages. It is imbalanced and historical; results should not be generalized to current messages or other populations. The dataset archive is included under the terms and attribution listed in [`data/README.md`](data/README.md).

**Dataset citation:** Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

## Method

- Multinomial Naive Bayes with Laplace smoothing (`alpha=1.0`), implemented with Python's standard library.
- Word unigram features compared with overlapping character n-grams of lengths 3, 4, and 5.
- Stratified five-fold evaluation with the same folds for both feature representations; each message receives an out-of-fold prediction.
- Pooled metrics, fold-to-fold mean and standard deviation, and descriptive short/long message groups.
- The length-group thresholds are computed from each fold's training data separately for spam and legitimate messages.

## Results

| Features | Accuracy | Spam recall | Legitimate-message false-positive rate | Missed spam | False spam flags |
|---|---:|---:|---:|---:|---:|
| Word unigrams | 98.1% | 94.9% | 1.4% | 38 | 69 |
| Character 3–5-grams | 98.4% | 95.0% | 1.1% | 37 | 51 |

The character model's pooled spam recall was 0.13 percentage points higher and its false-positive rate was 0.37 percentage points lower in this evaluation. These are small descriptive differences; this project does not test whether they are statistically significant.

### Important evaluation limitation

An exact-duplicate audit (case-folding and whitespace normalization only) found 290 duplicate groups in the corpus; 241 groups, containing 601 rows, appeared in more than one random stratified fold. This creates a risk that the model sees an exact normalized copy of a held-out message during training. The reported scores may therefore overstate performance on unseen messages. The next research step is to rerun the comparison with duplicate groups kept together and compare those results with the current row-stratified evaluation.

## Run it

Requires Python 3.10 or later. No third-party packages are required.

From this project folder, run:

```powershell
python main.py
```

If `python` is not recognized in PowerShell, use your installed Python executable, for example:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" .\main.py
```

The program writes the report and result tables under `report.md` and `results/`. It also creates row-level out-of-fold and misclassification files locally; those files contain message text and are excluded from this repository by `.gitignore`.

## Repository contents

- `main.py` — data loading, feature extraction, model fitting, cross-validation, duplicate audit, metrics, and report generation.
- `data/sms-spam-collection.zip` — the attributed UCI dataset archive.
- `data/README.md` — dataset source, citation, and license information.
- `report.md` — complete method, subgroup findings, limitations, and reproducibility notes.
- `results/model_comparison_metrics.csv` — pooled and fold-summary metrics.
- `results/length_group_metrics.csv` — short/long subgroup metrics.
- `results/fold_metrics.csv` — per-fold aggregate metrics.

Row-level predictions and misclassified messages are deliberately not committed; they contain the original SMS text.

## Limitations and AI assistance

This is an exploratory, AI-assisted student project, not a new algorithm or a claim of general superiority. It uses one historical, English-only corpus and a simple baseline classifier. The cross-fold duplicate finding means the existing row-stratified scores should be treated cautiously. A duplicate-grouped evaluation and an additional strong baseline would be needed before making stronger performance claims. Review the code, understand the method, and accurately describe your own contribution before presenting the project.