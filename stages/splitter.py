"""Train/validation split with optional stratification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path("artifacts")


def create_split(
    train_df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """Split training data into train and validation sets.

    Args:
        train_df: Preprocessed training DataFrame with text and label columns.
        config: Must contain random_seed and validation_split.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val, val_ids, split_report dict).

    Raises:
        OSError: If split_report.json cannot be written.
    """
    seed = config.get("random_seed", 42)
    if "random_seed" not in config:
        print("Warning: config missing random_seed; using default 42")
    val_split = config.get("validation_split", 0.2)
    if "validation_split" not in config:
        print("Warning: config missing validation_split; using default 0.2")

    X = train_df["text"]
    y = train_df["label"]
    ids = train_df["id"]
    stratified = True
    warnings: List[str] = []

    try:
        X_train, X_val, y_train, y_val, _, val_ids = train_test_split(
            X,
            y,
            ids,
            test_size=val_split,
            random_state=seed,
            stratify=y,
        )
    except ValueError as exc:
        stratified = False
        warnings.append(f"Stratified split failed ({exc}); using non-stratified split")
        print(f"Warning: {warnings[-1]}")
        X_train, X_val, y_train, y_val, _, val_ids = train_test_split(
            X,
            y,
            ids,
            test_size=val_split,
            random_state=seed,
            stratify=None,
        )

    labels_counts: Dict[str, Dict[str, int]] = {}
    for label in sorted(y.unique(), key=str):
        labels_counts[str(label)] = {
            "train": int((y_train == label).sum()),
            "validation": int((y_val == label).sum()),
        }

    split_report: Dict[str, Any] = {
        "random_seed": seed,
        "validation_split": val_split,
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "stratified": stratified,
        "labels": labels_counts,
    }
    if warnings:
        print(f"Split warnings: {warnings}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "split_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(split_report, fh, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write {report_path}: {exc}") from exc

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        val_ids.reset_index(drop=True),
        split_report,
    )
