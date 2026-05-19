# Deriv Interview — Text Classification Pipeline

## Overview

This project is a replayable, CPU-only machine learning pipeline for text classification. It loads labeled training data from CSV, trains several scikit-learn models with TF-IDF features, selects a winner on a held-out validation split, and writes reproducible artifacts for inference and auditing.

## Project Structure

```
deriv-interview/
├── pipeline.py              # Main entry: runs all stages in order
├── predict.py               # Single-text and batch inference CLI
├── validate.py              # Artifact validation checker
├── config.json              # Pipeline hyperparameters and options
├── train.csv                # Labeled training data (id, text, label)
├── test.csv                 # Unlabeled test data (id, text)
├── requirements.txt         # Python dependencies
├── stages/
│   ├── data_loader.py       # Load and validate CSV inputs
│   ├── preprocessor.py      # Text normalization (shared with predict.py)
│   ├── splitter.py          # Stratified train/validation split
│   ├── features.py          # TF-IDF vectorization
│   ├── trainer.py           # Train logistic/SVM/Naive Bayes models
│   ├── evaluator.py         # Validation metrics and cross-validation
│   ├── selector.py          # Pick winning model by metric
│   ├── artifacts.py         # Error analysis, predictions, reports
│   └── safeguards.py        # Data-quality checks after split
└── artifacts/               # Generated outputs (created at runtime)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run the full pipeline

```bash
python pipeline.py
```

### Run inference on a single text

```bash
python predict.py --text "Your text here"
```

### Run batch inference

```bash
python predict.py --input-file new_data.csv --output-file predictions.csv
```

### Validate all artifacts

```bash
python validate.py
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| **INIT** | Create `artifacts/` and load `config.json` |
| **DATA_LOADED** | Begin data ingest |
| **DATA_VALIDATED** | Load `train.csv` / `test.csv` with schema checks |
| **TEXT_PREPROCESSED** | Lowercase, strip, collapse whitespace |
| **SPLIT_CREATED** | Stratified train/validation split + safeguards |
| **FEATURES_FIT** | Fit TF-IDF on training split only |
| **MODELS_TRAINED** | Train all models from config |
| **CROSS_VALIDATION** | Optional stratified k-fold macro-F1 (if configured) |
| **MODELS_EVALUATED** | Validation metrics and confusion matrices |
| **FEATURE_IMPORTANCE** | Optional top tokens per class (logistic regression) |
| **WINNER_SELECTED** | Choose best model by `selection_metric` |
| **ARTIFACTS_SAVED** | Misclassification error analysis |
| **TEST_PREDICTIONS_GENERATED** | Predict labels for `test.csv` |
| **REPORT_EXPORTED** | `run_manifest.json`, optional HTML report |

## Generated Artifacts

| Filename | Description | Required/Optional |
|----------|-------------|-------------------|
| `data_validation_report.json` | Load/validation summary | Required |
| `preprocessing_preview.json` | Sample before/after text | Required |
| `split_report.json` | Split sizes and label counts | Required |
| `vectorizer.joblib` | Fitted TF-IDF vectorizer | Required |
| `*.joblib` (models) | Fitted classifiers | Required |
| `metrics.json` | Per-model validation metrics | Required |
| `model_selection_report.json` | Winner and rationale | Required |
| `error_analysis.json` | Top misclassified validation rows | Required |
| `test_predictions.csv` | Predictions for test set | Required |
| `safeguards_report.json` | Class balance / leakage warnings | Required |
| `run_manifest.json` | Run metadata and artifact list | Required |
| `cross_validation_report.json` | CV mean/std per model | Optional |
| `feature_importance.json` | Top TF-IDF tokens per class | Optional |
| `report.html` | Human-readable summary | Optional |
| `pipeline.log` | Pipeline log file | Optional |

## Configuration

| Key | Type | Description |
|-----|------|-------------|
| `random_seed` | int | Seed for splits, models, and CV |
| `validation_split` | float | Fraction of training data for validation (e.g. `0.2`) |
| `cross_validation_folds` | int | If set, run stratified k-fold CV before validation eval |
| `models` | list[str] | Model names to train: `logistic_regression`, `linear_svm`, `naive_bayes` |
| `vectorizer.type` | str | Feature type (`tfidf`) |
| `vectorizer.ngram_range` | list[int] | Min/max n-gram sizes, e.g. `[1, 2]` |
| `vectorizer.max_features` | int | Maximum vocabulary size |
| `vectorizer.min_df` | int | Minimum document frequency for terms |
| `selection_metric` | str | Metric to pick winner: `macro_f1`, `macro_precision`, etc. |
| `top_k_error_examples` | int | Max misclassified examples in error analysis |
| `feature_importance` | bool | Enable top-token report (default: `true`) |
| `feature_importance_top_k` | int | Tokens per class in importance report (default: `20`) |
| `enable_file_logging` | bool | Write `artifacts/pipeline.log` (default: `true`) |
| `enable_stage_timings` | bool | Record per-stage seconds in manifest (default: `true`) |
| `html_report` | bool | Generate `artifacts/report.html` (default: `true`) |

## Design Decisions

- **TF-IDF with (1,2)-grams** — Strong baseline for short text; fast on CPU, interpretable, and no GPU or external APIs.
- **Three linear models** — Logistic regression, linear SVM, and multinomial Naive Bayes cover complementary assumptions while staying lightweight.
- **Stratified validation split** — Keeps class proportions stable on small datasets; falls back to random split with a warning if stratification is impossible.
- **Disk-backed inference** — `predict.py` and test prediction reload `vectorizer.joblib` and the winner from disk to prove artifacts are self-contained.

## Reproducibility

Use a fixed `random_seed` in `config.json`, keep the same `train.csv` and `test.csv`, and delete `artifacts/` before each full run. The same seed yields the same split, model fits (given identical library versions), and winner selection.

## CPU Requirements

Runs on any laptop CPU. No GPU is required. A typical full pipeline run completes in under 30 seconds on standard hardware.
