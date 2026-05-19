"""Evaluate trained models on the validation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from stages.features import transform
from stages.trainer import instantiate_model

ARTIFACTS_DIR = Path("artifacts")


def _confusion_matrix_dict(y_true, y_pred, labels: List[str]) -> Dict[str, Dict[str, int]]:
    """Build nested confusion matrix dict {true: {pred: count}}.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of label values for matrix axes.

    Returns:
        Nested dictionary mapping true label to predicted label counts.

    Raises:
        None.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result: Dict[str, Dict[str, int]] = {}
    for i, true_lbl in enumerate(labels):
        result[str(true_lbl)] = {str(labels[j]): int(cm[i, j]) for j in range(len(labels))}
    return result


def run_cross_validation(
    models_config: List[str],
    vectorizer,
    X_train: pd.Series,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Run stratified k-fold cross-validation with macro F1 scoring per model.

    Args:
        models_config: List of model names to evaluate (e.g. config["models"]).
        vectorizer: Fitted TfidfVectorizer used to transform X_train.
        X_train: Training texts (preprocessed).
        y_train: Training labels aligned with X_train.
        config: Pipeline config with cross_validation_folds and random_seed.

    Returns:
        Dictionary keyed by model name with mean, std, and per_fold_scores.

    Raises:
        OSError: If cross_validation_report.json cannot be written.
        ValueError: If cross_validation_folds is missing or invalid.
    """
    n_folds = config.get("cross_validation_folds")
    if n_folds is None:
        raise ValueError("config must contain cross_validation_folds for cross-validation")
    seed = config.get("random_seed", 42)
    if "random_seed" not in config:
        print("Warning: config missing random_seed; using default 42")

    X_features = transform(vectorizer, X_train)
    cv = StratifiedKFold(
        n_splits=int(n_folds),
        shuffle=True,
        random_state=seed,
    )

    report: Dict[str, Dict[str, Any]] = {}
    for name in models_config:
        model = instantiate_model(name, seed)
        fold_scores = cross_val_score(
            model,
            X_features,
            y_train,
            cv=cv,
            scoring="f1_macro",
        )
        report[name] = {
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
            "per_fold_scores": [float(s) for s in fold_scores],
        }
        print(
            f"Cross-validation {name}: mean f1_macro={report[name]['mean']:.4f} "
            f"(+/- {report[name]['std']:.4f})"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "cross_validation_report.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write {path}: {exc}") from exc

    return report


def evaluate_models(
    models: Dict[str, Any],
    vectorizer,
    X_val: pd.Series,
    y_val: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Evaluate each model on validation data and save metrics.json.

    Args:
        models: Map of model name to fitted estimator.
        vectorizer: Fitted TfidfVectorizer.
        X_val: Validation texts.
        y_val: Validation labels.
        config: Pipeline configuration.

    Returns:
        Nested metrics dictionary keyed by model name.

    Raises:
        OSError: If metrics.json cannot be written.
    """
    _ = config
    X_val_features = transform(vectorizer, X_val)
    labels = sorted(y_val.unique(), key=str)
    metrics: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        y_pred = model.predict(X_val_features)
        per_class_precision = precision_score(
            y_val, y_pred, labels=labels, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            y_val, y_pred, labels=labels, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            y_val, y_pred, labels=labels, average=None, zero_division=0
        )
        per_class = {}
        for i, lbl in enumerate(labels):
            per_class[str(lbl)] = {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            }

        metrics[name] = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "macro_precision": float(
                precision_score(y_val, y_pred, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_val, y_pred, average="macro", zero_division=0)
            ),
            "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
            "per_class": per_class,
            "confusion_matrix": _confusion_matrix_dict(y_val, y_pred, labels),
        }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "metrics.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write {path}: {exc}") from exc

    return metrics


def extract_feature_importance(
    models: Dict[str, Any],
    vectorizer,
    label_names: List[str],
    config: Dict[str, Any],
) -> None:
    """Extract and save top predictive tokens per class for interpretable models.

    Args:
        models: Dictionary mapping model name to fitted sklearn estimator.
        vectorizer: Fitted TfidfVectorizer used during training.
        label_names: List of class label strings in model coefficient order.
        config: Pipeline configuration dictionary.

    Returns:
        None. Saves ``artifacts/feature_importance.json`` on success.

    Raises:
        None. Logs a warning and returns on failure or unsupported models.
    """
    if not config.get("feature_importance", True):
        return
    try:
        if "logistic_regression" not in models:
            return
        model = models["logistic_regression"]
        if not hasattr(model, "coef_"):
            print("Warning: logistic_regression has no coef_; skipping feature importance")
            return

        top_k = int(config.get("feature_importance_top_k", 20))
        coef = model.coef_
        if hasattr(vectorizer, "get_feature_names_out"):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()

        top_features_per_class: Dict[str, List[str]] = {}
        for idx, label in enumerate(label_names):
            if idx >= len(coef):
                break
            row = coef[idx]
            top_indices = np.argsort(row)[-top_k:][::-1]
            top_features_per_class[str(label)] = [
                str(feature_names[i]) for i in top_indices
            ]

        payload = {
            "model": "logistic_regression",
            "top_features_per_class": top_features_per_class,
        }
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS_DIR / "feature_importance.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
    except Exception as exc:
        print(f"Warning: feature importance extraction failed: {exc}")
