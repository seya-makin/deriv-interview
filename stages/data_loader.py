"""Load and validate train/test CSV data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
REQUIRED_TRAIN_COLS = ["id", "text", "label"]
REQUIRED_TEST_COLS = ["id", "text"]


def _project_root() -> Path:
    """Return the project root directory (parent of stages package).

    Args:
        None.

    Returns:
        Path to the repository root.

    Raises:
        None.
    """
    return Path(__file__).resolve().parent.parent


def _save_validation_report(report: Dict[str, Any]) -> None:
    """Persist the validation report to artifacts.

    Args:
        report: Validation report dictionary to serialize.

    Returns:
        None.

    Raises:
        SystemExit: If the report file cannot be written.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "data_validation_report.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    except OSError as exc:
        print(f"Failed to write validation report to {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _fail(message: str, report: Dict[str, Any]) -> None:
    """Mark report as failed, save it, and exit.

    Args:
        message: Error message printed to stderr.
        report: Mutable validation report updated to failed status.

    Returns:
        None.

    Raises:
        SystemExit: Always exits with code 1.
    """
    report["status"] = "failed"
    _save_validation_report(report)
    print(message, file=sys.stderr)
    sys.exit(1)


def load_and_validate(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load train/test CSV files, validate schema and content, return cleaned frames.

    Args:
        config: Pipeline configuration dictionary (unused for paths; kept for API consistency).

    Returns:
        Tuple of (train_df, test_df, validation_report dict).

    Raises:
        SystemExit: On validation failure or unreadable input files.
    """
    _ = config
    root = _project_root()
    train_path = root / "train.csv"
    test_path = root / "test.csv"

    report: Dict[str, Any] = {
        "status": "passed",
        "train_rows": 0,
        "test_rows": 0,
        "distinct_labels": [],
        "empty_text_rows_dropped": 0,
        "duplicate_ids_train": [],
        "duplicate_ids_test": [],
        "warnings": [],
    }

    try:
        train_df = pd.read_csv(train_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        _fail(f"Failed to read train.csv from {train_path}: {exc}", report)

    try:
        test_df = pd.read_csv(test_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        _fail(f"Failed to read test.csv from {test_path}: {exc}", report)

    missing_train = [c for c in REQUIRED_TRAIN_COLS if c not in train_df.columns]
    if missing_train:
        _fail(
            f"train.csv must have columns {REQUIRED_TRAIN_COLS}; "
            f"missing {missing_train}, found {list(train_df.columns)}",
            report,
        )

    if not all(c in test_df.columns for c in REQUIRED_TEST_COLS):
        _fail(
            f"test.csv must have columns {REQUIRED_TEST_COLS}; "
            f"found {list(test_df.columns)}",
            report,
        )

    train_df, train_dropped = _clean_text_column(train_df, "train", report)
    test_df, test_dropped = _clean_text_column(test_df, "test", report)
    report["empty_text_rows_dropped"] = train_dropped + test_dropped

    if train_df["label"].nunique() < 2:
        _fail(
            f"train.csv must have at least 2 distinct labels; "
            f"found {train_df['label'].nunique()}",
            report,
        )

    report["distinct_labels"] = sorted(train_df["label"].astype(str).unique().tolist())

    dup_train = _find_duplicate_ids(train_df)
    dup_test = _find_duplicate_ids(test_df)
    report["duplicate_ids_train"] = dup_train
    report["duplicate_ids_test"] = dup_test

    if dup_train:
        _fail(f"Duplicate ids in train.csv: {dup_train}", report)
    if dup_test:
        _fail(f"Duplicate ids in test.csv: {dup_test}", report)

    report["train_rows"] = len(train_df)
    report["test_rows"] = len(test_df)
    report["status"] = "passed"
    _save_validation_report(report)

    return train_df, test_df, report


def _find_duplicate_ids(df: pd.DataFrame) -> List[Any]:
    """Return list of id values that appear more than once.

    Args:
        df: DataFrame containing an ``id`` column.

    Returns:
        Sorted list of duplicate id values.

    Raises:
        None.
    """
    duplicated = df[df["id"].duplicated(keep=False)]
    if duplicated.empty:
        return []
    return sorted(duplicated["id"].unique().tolist(), key=str)


def _clean_text_column(
    df: pd.DataFrame, name: str, report: Dict[str, Any]
) -> Tuple[pd.DataFrame, int]:
    """Drop or fix invalid text rows; return cleaned frame and drop count.

    Args:
        df: Input DataFrame with ``text`` column.
        name: Dataset name used in warning messages (train or test).
        report: Validation report dict receiving warning strings.

    Returns:
        Tuple of (cleaned DataFrame, number of rows dropped).

    Raises:
        None.
    """
    df = df.copy()
    dropped = 0
    mask_keep = []

    for idx, row in df.iterrows():
        text = row["text"]
        if pd.isna(text):
            report["warnings"].append(f"{name} row index {idx}: text is null/NaN, dropping")
            dropped += 1
            mask_keep.append(False)
            continue
        if not isinstance(text, str):
            report["warnings"].append(
                f"{name} row index {idx}: text is not str ({type(text).__name__}), "
                "attempting cast"
            )
            try:
                text = str(text)
            except Exception:
                report["warnings"].append(f"{name} row index {idx}: could not cast text, dropping")
                dropped += 1
                mask_keep.append(False)
                continue
            df.at[idx, "text"] = text
        if not text.strip():
            report["warnings"].append(f"{name} row id {row.get('id', idx)}: empty text after strip, dropping")
            dropped += 1
            mask_keep.append(False)
            continue
        mask_keep.append(True)

    cleaned = df[mask_keep].reset_index(drop=True)
    return cleaned, dropped
