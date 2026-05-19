"""Text preprocessing for train, test, and inference."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """Apply the canonical preprocessing steps to a single text string.

    Steps: lowercase, strip edges, collapse internal whitespace; guard null/empty.

    Args:
        text: Raw input text (may be None in callers that pass str only).

    Returns:
        Preprocessed text string (empty string if input is invalid).

    Raises:
        None.
    """
    if text is None:
        print("Warning: preprocess_text received None; replacing with empty string", file=sys.stderr)
        return ""
    if not isinstance(text, str):
        print(
            f"Warning: preprocess_text expected str, got {type(text).__name__}; casting",
            file=sys.stderr,
        )
        text = str(text)
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    if not text:
        print("Warning: text empty after preprocessing", file=sys.stderr)
    return text


def preprocess(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Preprocess text columns in train and test DataFrames identically.

    Args:
        train_df: Training DataFrame with id, text, label columns.
        test_df: Test DataFrame with id, text columns.
        config: Pipeline configuration (unused; kept for API consistency).

    Returns:
        Tuple of (processed train_df, processed test_df, preview list for first 5 rows).

    Raises:
        OSError: If the preprocessing preview artifact cannot be written.
    """
    _ = config
    train_out = train_df.copy()
    test_out = test_df.copy()

    preview: List[Dict[str, Any]] = []
    limit = 5

    for df in (train_out, test_out):
        originals = df["text"].tolist()
        df["text"] = [preprocess_text(t) for t in originals]
        if len(preview) < limit:
            for i in range(min(limit - len(preview), len(df))):
                row = df.iloc[i]
                preview.append(
                    {
                        "id": str(row["id"]),
                        "original": str(originals[i]),
                        "processed": str(row["text"]),
                    }
                )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = ARTIFACTS_DIR / "preprocessing_preview.json"
    try:
        with open(preview_path, "w", encoding="utf-8") as fh:
            json.dump(preview, fh, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write {preview_path}: {exc}") from exc

    return train_out, test_out, preview
