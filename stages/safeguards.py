"""Data-quality safeguards after splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
IMBALANCE_THRESHOLD = 0.8


def run_safeguards(
    train_df: pd.DataFrame,
    X_train: pd.Series,
    X_val: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run class-balance, label-coverage, and duplicate-text checks.

    Args:
        train_df: Full training DataFrame (unused except API consistency).
        X_train: Training texts after split.
        X_val: Validation texts after split.
        y_train: Training labels.
        y_val: Validation labels.
        config: Pipeline configuration (unused).

    Returns:
        Safeguards report dictionary.

    Raises:
        OSError: If safeguards_report.json cannot be written.
    """
    _ = train_df, config
    warnings: List[str] = []
    train_counts = y_train.value_counts(normalize=True)
    class_proportions = {str(k): float(v) for k, v in train_counts.items()}
    class_imbalance_warning = any(p > IMBALANCE_THRESHOLD for p in train_counts)
    if class_imbalance_warning:
        warnings.append(
            f"Class imbalance: at least one class exceeds {IMBALANCE_THRESHOLD:.0%} of training set"
        )

    train_labels = set(y_train.unique())
    val_labels = set(y_val.unique())
    missing_val_classes = sorted(str(l) for l in train_labels - val_labels)
    if missing_val_classes:
        warnings.append(f"Training labels missing from validation: {missing_val_classes}")

    train_texts = set(X_train.astype(str))
    val_texts = set(X_val.astype(str))
    duplicate_count = len(train_texts & val_texts)
    if duplicate_count > 0:
        warnings.append(
            f"Found {duplicate_count} exact duplicate text(s) shared between train and validation"
        )

    report: Dict[str, Any] = {
        "class_imbalance_warning": bool(class_imbalance_warning),
        "class_proportions": class_proportions,
        "missing_val_classes": missing_val_classes,
        "train_val_duplicate_texts": int(duplicate_count),
        "warnings": warnings,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "safeguards_report.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write {path}: {exc}") from exc

    for w in warnings:
        print(f"Safeguard warning: {w}")

    return report
