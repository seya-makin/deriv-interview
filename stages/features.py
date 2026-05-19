"""TF-IDF feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

ARTIFACTS_DIR = Path("artifacts")


def _build_vectorizer(config: Dict[str, Any]) -> TfidfVectorizer:
    """Construct TfidfVectorizer from config vectorizer section.

    Args:
        config: Pipeline configuration with optional ``vectorizer`` block.

    Returns:
        Unfitted TfidfVectorizer instance.

    Raises:
        None.
    """
    vec_cfg = config.get("vectorizer", {})
    if "vectorizer" not in config:
        print("Warning: config missing vectorizer; using defaults")
    ngram = vec_cfg.get("ngram_range", [1, 2])
    if isinstance(ngram, list):
        ngram = tuple(ngram)
    return TfidfVectorizer(
        ngram_range=ngram,
        max_features=vec_cfg.get("max_features", 5000),
        min_df=vec_cfg.get("min_df", 1),
    )


def fit_vectorizer(
    X_train: pd.Series, config: Dict[str, Any]
) -> Tuple[TfidfVectorizer, spmatrix, Dict[str, Any]]:
    """Fit TF-IDF vectorizer on training texts only.

    Args:
        X_train: Training text series.
        config: Pipeline configuration with vectorizer settings.

    Returns:
        Tuple of (fitted vectorizer, transformed X_train sparse matrix, config).

    Raises:
        OSError: If vectorizer cannot be saved to disk.
    """
    vectorizer = _build_vectorizer(config)
    X_train_features = vectorizer.fit_transform(X_train.astype(str))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "vectorizer.joblib"
    try:
        joblib.dump(vectorizer, path)
    except OSError as exc:
        raise OSError(f"Failed to save vectorizer to {path}: {exc}") from exc
    print(f"Vectorizer fitted and saved to {path}")
    return vectorizer, X_train_features, config


def transform(vectorizer: TfidfVectorizer, X: Union[pd.Series, List[str]]) -> spmatrix:
    """Transform raw texts using a fitted vectorizer (no refit).

    Args:
        vectorizer: Fitted TfidfVectorizer.
        X: Text series or list of strings.

    Returns:
        Sparse feature matrix.

    Raises:
        None.
    """
    if isinstance(X, pd.Series):
        texts = X.astype(str)
    else:
        texts = [str(t) for t in X]
    return vectorizer.transform(texts)
