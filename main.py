"""Compare word and character n-gram features with stratified five-fold evaluation."""

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import random
import re
import statistics
import zipfile

ROOT = Path(__file__).resolve().parent
ZIP_FILE = ROOT / "data" / "sms-spam-collection.zip"
RESULTS = ROOT / "results"
RANDOM_SEED = 42
N_FOLDS = 5
ALPHA = 1.0
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
MODEL_NAMES = {
    "word": "Word unigrams",
    "char": "Character 3-5 grams",
}
GROUP_NAMES = ("all", "spam_short", "spam_long", "ham_short", "ham_long")


def read_messages():
    """Read the tab-separated labeled SMS file from the UCI ZIP archive."""
    with zipfile.ZipFile(ZIP_FILE) as archive:
        member = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
        content = archive.read(member).decode("utf-8")
    messages = []
    for line in content.splitlines():
        label, separator, text = line.partition("\t")
        if separator and label in {"ham", "spam"}:
            messages.append((label, text))
    return messages


def make_stratified_folds(rows, n_folds=N_FOLDS):
    """Create reproducible folds with similar ham/spam proportions in each fold."""
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[0]].append(row)

    rng = random.Random(RANDOM_SEED)
    folds = [[] for _ in range(n_folds)]
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        for index, row in enumerate(group):
            folds[index % n_folds].append(row)
    for fold in folds:
        rng.shuffle(fold)
    return folds


def normalize_message(text):
    """Normalize only case and whitespace for an exact-duplicate audit."""
    return " ".join(text.casefold().split())


def audit_duplicates(messages, folds):
    """Count exact normalized duplicate groups and groups crossing CV folds."""
    counts = Counter(normalize_message(text) for _, text in messages)
    labels_by_text = defaultdict(set)
    fold_ids_by_text = defaultdict(set)
    for label, text in messages:
        labels_by_text[normalize_message(text)].add(label)
    for fold_number, fold in enumerate(folds, start=1):
        for _, text in fold:
            fold_ids_by_text[normalize_message(text)].add(fold_number)

    duplicate_texts = {text for text, count in counts.items() if count > 1}
    crossing_texts = {
        text for text in duplicate_texts if len(fold_ids_by_text[text]) > 1
    }
    return {
        "duplicate_groups": len(duplicate_texts),
        "duplicate_rows_beyond_first": sum(counts[text] - 1 for text in duplicate_texts),
        "conflicting_label_groups": sum(len(labels) > 1 for labels in labels_by_text.values()),
        "duplicate_groups_crossing_folds": len(crossing_texts),
        "rows_in_duplicate_groups_crossing_folds": sum(counts[text] for text in crossing_texts),
    }


def extract_features(text, representation):
    """Represent text as word tokens or overlapping character n-grams."""
    if representation == "word":
        return TOKEN_RE.findall(text.lower())
    normalized = " ".join(text.lower().split())
    padded = f" {normalized} "
    return [padded[i:i + size] for size in (3, 4, 5) for i in range(len(padded) - size + 1)]


def fit_naive_bayes(training_rows, representation):
    """Learn feature counts for each class using a Multinomial Naive Bayes model."""
    feature_counts = {"ham": Counter(), "spam": Counter()}
    feature_totals = Counter()
    message_totals = Counter()
    vocabulary = set()

    for label, text in training_rows:
        features = extract_features(text, representation)
        feature_counts[label].update(features)
        feature_totals[label] += len(features)
        message_totals[label] += 1
        vocabulary.update(features)

    return feature_counts, feature_totals, message_totals, vocabulary, representation


def predict(text, model):
    """Return the class with the larger smoothed log probability score."""
    feature_counts, feature_totals, message_totals, vocabulary, representation = model
    vocab_size = max(1, len(vocabulary))
    total_messages = sum(message_totals.values())
    scores = {}

    for label in ("ham", "spam"):
        score = math.log(message_totals[label] / total_messages)
        denominator = feature_totals[label] + ALPHA * vocab_size
        for feature, count in Counter(extract_features(text, representation)).items():
            probability = (feature_counts[label][feature] + ALPHA) / denominator
            score += count * math.log(probability)
        scores[label] = score
    return "spam" if scores["spam"] > scores["ham"] else "ham"


def calculate_metrics(rows):
    """Compute metrics from actual and predicted labels."""
    counts = Counter((row["actual"], row["predicted"]) for row in rows)
    spam_total = counts[("spam", "spam")] + counts[("spam", "ham")]
    ham_total = counts[("ham", "ham")] + counts[("ham", "spam")]
    predicted_spam = counts[("spam", "spam")] + counts[("ham", "spam")]
    true_spam = counts[("spam", "spam")]
    total = spam_total + ham_total
    return {
        "messages": total,
        "spam_messages": spam_total,
        "ham_messages": ham_total,
        "accuracy": (true_spam + counts[("ham", "ham")]) / total if total else None,
        "spam_recall": true_spam / spam_total if spam_total else None,
        "ham_false_positive_rate": counts[("ham", "spam")] / ham_total if ham_total else None,
        "spam_precision": true_spam / predicted_spam if predicted_spam else None,
        "false_negatives": counts[("spam", "ham")],
        "false_positives": counts[("ham", "spam")],
    }


def select_group(rows, group_name):
    if group_name == "all":
        return rows
    label, length_group = group_name.split("_")
    return [row for row in rows if row["actual"] == label and row["length_group"] == length_group]


def mean_or_none(values):
    available = [value for value in values if value is not None]
    return statistics.mean(available) if available else None


def stdev_or_none(values):
    available = [value for value in values if value is not None]
    return statistics.stdev(available) if len(available) > 1 else None


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def show(value):
    return "N/A" if value is None else f"{value:.3f}"


def show_mean_sd(mean, sd):
    if mean is None:
        return "N/A"
    if sd is None:
        return show(mean)
    return f"{mean:.3f} ± {sd:.4f}"


def percentage_points(value):
    return f"{value * 100:+.2f} percentage points"


def main():
    if not ZIP_FILE.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_FILE}")
    messages = read_messages()
    folds = make_stratified_folds(messages)
    duplicate_audit = audit_duplicates(messages, folds)
    all_predictions = {representation: [] for representation in MODEL_NAMES}
    fold_metrics = []
    median_records = []

    for fold_number, test_rows in enumerate(folds, start=1):
        test_ids = {id(row) for row in test_rows}
        training_rows = [row for row in messages if id(row) not in test_ids]
        # Use only training texts to calculate cutoffs; each held-out fold remains unseen.
        class_medians = {
            label: statistics.median(len(text.split()) for actual, text in training_rows if actual == label)
            for label in ("ham", "spam")
        }
        median_records.append({"fold": fold_number, **class_medians})

        for representation, model_name in MODEL_NAMES.items():
            model = fit_naive_bayes(training_rows, representation)
            predictions = []
            for actual, text in test_rows:
                length = len(text.split())
                predictions.append({
                    "model": model_name,
                    "fold": fold_number,
                    "actual": actual,
                    "predicted": predict(text, model),
                    "words": length,
                    "length_group": "short" if length <= class_medians[actual] else "long",
                    "message": text,
                })
            all_predictions[representation].extend(predictions)
            for group_name in GROUP_NAMES:
                selected = select_group(predictions, group_name)
                fold_metrics.append({
                    "model": model_name,
                    "fold": fold_number,
                    "group": group_name,
                    **calculate_metrics(selected),
                })

    summary_metrics = []
    for representation, model_name in MODEL_NAMES.items():
        predictions = all_predictions[representation]
        for group_name in GROUP_NAMES:
            pooled = calculate_metrics(select_group(predictions, group_name))
            per_fold = [row for row in fold_metrics if row["model"] == model_name and row["group"] == group_name]
            summary = {"model": model_name, "group": group_name, **pooled}
            for metric in ("accuracy", "spam_recall", "ham_false_positive_rate"):
                values = [row[metric] for row in per_fold]
                summary[f"{metric}_fold_mean"] = mean_or_none(values)
                summary[f"{metric}_fold_sd"] = stdev_or_none(values)
            summary_metrics.append(summary)

    RESULTS.mkdir(exist_ok=True)
    summary_columns = [
        "model", "group", "messages", "spam_messages", "ham_messages", "accuracy",
        "accuracy_fold_mean", "accuracy_fold_sd", "spam_recall", "spam_recall_fold_mean",
        "spam_recall_fold_sd", "ham_false_positive_rate", "ham_false_positive_rate_fold_mean",
        "ham_false_positive_rate_fold_sd", "spam_precision", "false_negatives", "false_positives",
    ]
    write_csv(RESULTS / "model_comparison_metrics.csv", summary_metrics, summary_columns)
    write_csv(RESULTS / "length_group_metrics.csv", [row for row in summary_metrics if row["group"] != "all"], summary_columns)
    fold_columns = [
        "model", "fold", "group", "messages", "spam_messages", "ham_messages", "accuracy",
        "spam_recall", "ham_false_positive_rate", "spam_precision", "false_negatives", "false_positives",
    ]
    write_csv(RESULTS / "fold_metrics.csv", fold_metrics, fold_columns)

    pooled_predictions = [row for rows in all_predictions.values() for row in rows]
    prediction_columns = ["model", "fold", "actual", "predicted", "words", "length_group", "message"]
    write_csv(RESULTS / "out_of_fold_predictions.csv", pooled_predictions, prediction_columns)
    mistakes = [row for row in pooled_predictions if row["actual"] != row["predicted"]]
    write_csv(RESULTS / "misclassified_messages.csv", mistakes, prediction_columns)

    def summary_for(representation, group):
        name = MODEL_NAMES[representation]
        return next(row for row in summary_metrics if row["model"] == name and row["group"] == group)

    overall_word = summary_for("word", "all")
    overall_char = summary_for("char", "all")
    recall_change = overall_char["spam_recall"] - overall_word["spam_recall"]
    fpr_change = overall_char["ham_false_positive_rate"] - overall_word["ham_false_positive_rate"]

    overall_lines = []
    for representation in MODEL_NAMES:
        row = summary_for(representation, "all")
        overall_lines.append(
            f"| {row['model']} | {row['messages']} | {show(row['accuracy'])} | "
            f"{show_mean_sd(row['accuracy_fold_mean'], row['accuracy_fold_sd'])} | "
            f"{show(row['spam_recall'])} | {show_mean_sd(row['spam_recall_fold_mean'], row['spam_recall_fold_sd'])} | "
            f"{show(row['ham_false_positive_rate'])} | "
            f"{show_mean_sd(row['ham_false_positive_rate_fold_mean'], row['ham_false_positive_rate_fold_sd'])} | "
            f"{row['false_negatives']} | {row['false_positives']} |"
        )

    spam_lines, ham_lines = [], []
    for representation in MODEL_NAMES:
        for group in ("spam_short", "spam_long"):
            row = summary_for(representation, group)
            spam_lines.append(
                f"| {row['model']} | {group.replace('_', ' ')} | {row['messages']} | "
                f"{show(row['spam_recall'])} | {show_mean_sd(row['spam_recall_fold_mean'], row['spam_recall_fold_sd'])} | "
                f"{row['false_negatives']} |"
            )
        for group in ("ham_short", "ham_long"):
            row = summary_for(representation, group)
            ham_lines.append(
                f"| {row['model']} | {group.replace('_', ' ')} | {row['messages']} | "
                f"{show(row['ham_false_positive_rate'])} | "
                f"{show_mean_sd(row['ham_false_positive_rate_fold_mean'], row['ham_false_positive_rate_fold_sd'])} | "
                f"{row['false_positives']} |"
            )

    ham_cutoffs = {row["ham"] for row in median_records}
    spam_cutoffs = {row["spam"] for row in median_records}
    if len(ham_cutoffs) == 1 and len(spam_cutoffs) == 1:
        median_text = f"ham {next(iter(ham_cutoffs)):g} words and spam {next(iter(spam_cutoffs)):g} words in all folds"
    else:
        median_text = (
            f"ham {min(ham_cutoffs):g}-{max(ham_cutoffs):g} words and "
            f"spam {min(spam_cutoffs):g}-{max(spam_cutoffs):g} words across folds"
        )
    report = f'''# Word vs. character features for SMS spam detection

## Abstract
This exploratory study compares word unigram and character 3–5-gram representations for SMS spam classification. Both use the same Multinomial Naive Bayes classifier and stratified five-fold evaluation on the UCI SMS Spam Collection. Results are reported as pooled out-of-fold metrics and fold-to-fold variation. The comparison is descriptive and limited to this dataset.

## Research question
Does a word-based or character-based representation change spam detection performance, including error rates on shorter and longer messages?

## Dataset
The UCI SMS Spam Collection contains {len(messages):,} labeled SMS messages. It is English-language, historical, and class-imbalanced; it should not be treated as representative of current messages or all populations.

Source: UCI Machine Learning Repository, [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection), DOI: 10.24432/C5CC84.

## Experimental design
- **Evaluation:** stratified five-fold cross-validation with random seed {RANDOM_SEED}. Each message is held out for evaluation once. The same folds are used for both representations.
- **Classifier:** Multinomial Naive Bayes with Laplace smoothing (alpha={ALPHA}); implemented with Python's standard library.
- **Representations:** word unigrams; overlapping character n-grams of lengths 3, 4, and 5.
- **Length groups:** message length is whitespace-separated word count. Within each fold, training data set separate median cutoffs for ham and spam. Held-out messages are grouped using the cutoff for their true class. This keeps the length comparisons within class and avoids using held-out text to choose the cutoff.
- **Metrics:** spam recall (fraction of spam caught), ham false-positive rate (fraction of legitimate messages wrongly flagged), and accuracy. Pooled metrics combine all out-of-fold predictions; fold mean and sample standard deviation show variation across the five folds. The standard deviation is descriptive, not a confidence interval.

Training-set class-specific median cutoffs by fold: {median_text}.

## Exact-duplicate audit
Messages were normalized by case-folding and collapsing whitespace; no fuzzy duplicate matching was used. The audit found {duplicate_audit['duplicate_groups']} exact normalized duplicate groups, representing {duplicate_audit['duplicate_rows_beyond_first']} repeated rows beyond the first copy. {duplicate_audit['duplicate_groups_crossing_folds']} duplicate groups ({duplicate_audit['rows_in_duplicate_groups_crossing_folds']} rows) appeared in more than one random stratified fold. Therefore, an evaluation message may have an exact normalized copy in its training fold, and the reported cross-validation scores may be optimistic for unseen messages. This version does not include duplicate-grouped cross-validation; keeping duplicate groups together is an important next evaluation step.

## Results: overall

| Representation | Out-of-fold n | Pooled accuracy | Accuracy mean ± SD | Pooled spam recall | Spam recall mean ± SD | Pooled ham false-positive rate | Ham false-positive mean ± SD | Missed spam | False spam flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(overall_lines)}

## Results: length subgroups

### Spam recall
| Model | Group | n | Pooled recall | Fold mean ± SD | Missed spam |
|---|---|---:|---:|---:|---:|
{chr(10).join(spam_lines)}

### False-positive rate among legitimate messages
| Model | Group | n | Pooled false-positive rate | Fold mean ± SD | False spam flags |
|---|---|---:|---:|---:|---:|
{chr(10).join(ham_lines)}

## Interpretation
Compared with the word model, the character model's pooled spam recall changed by **{percentage_points(recall_change)}** and its pooled ham false-positive rate changed by **{percentage_points(fpr_change)}**. Positive recall change means more spam was caught; positive false-positive change means more legitimate messages were wrongly flagged. Consider pooled results together with fold variation and the different costs of these two error types. The experiment does not establish that either representation is generally superior.

## Limitations
The study uses one old, English-only dataset. Cross-validation measures variation across folds from this dataset; it does not test transfer to newer messages or different populations. Fold standard deviations are not formal confidence intervals. Naive Bayes is a simple baseline, and neither representation understands message meaning or context. A follow-up could evaluate a newer dataset and pre-register a broader set of metrics.

## Reproducibility
Run `python main.py` from this folder. The program writes pooled summary metrics to `results/model_comparison_metrics.csv`, length-group results to `results/length_group_metrics.csv`, per-fold metrics to `results/fold_metrics.csv`, all held-out predictions to `results/out_of_fold_predictions.csv`, and errors to `results/misclassified_messages.csv`.
'''
    (ROOT / "report.md").write_text(report, encoding="utf-8")

    print(f"Loaded {len(messages):,} messages; evaluated each message once in {N_FOLDS} folds.")
    print(f"Exact duplicate groups crossing folds: {duplicate_audit['duplicate_groups_crossing_folds']}; rows in those groups: {duplicate_audit['rows_in_duplicate_groups_crossing_folds']}.")
    print("Pooled results (out-of-fold predictions):")
    for representation in MODEL_NAMES:
        row = summary_for(representation, "all")
        print(f"  {row['model']}: accuracy={show(row['accuracy'])}, "
              f"spam recall={show(row['spam_recall'])}, ham false-positive rate={show(row['ham_false_positive_rate'])}")
    print(f"Character-minus-word spam recall: {percentage_points(recall_change)}")
    print(f"Character-minus-word ham false-positive rate: {percentage_points(fpr_change)}")
    print(f"Report: {ROOT / 'report.md'}")


if __name__ == "__main__":
    main()


