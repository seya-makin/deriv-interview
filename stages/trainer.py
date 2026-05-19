"""Train classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ARTIFACTS_DIR = Path("artifacts")


def instantiate_model(name: str, seed: int):
    """Create an sklearn classifier by configured name.

    Args:
        name: Model identifier from config (e.g. logistic_regression).
        seed: Random seed for supported estimators.

    Returns:
        Unfitted sklearn estimator.

    Raises:
        ValueError: If name is not a supported model.
    """
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, max_iter=1000, C=1.0)
    if name == "linear_svm":
        return LinearSVC(random_state=seed, max_iter=1000, C=1.0)
    if name == "naive_bayes":
        return MultinomialNB(alpha=1.0)
    raise ValueError(f"Unknown model name: {name}")


def train_models(
    X_train_features: spmatrix, y_train, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Train all models listed in config and persist each to artifacts.

    Args:
        X_train_features: Sparse training feature matrix.
        y_train: Training labels.
        config: Pipeline config with models list and random_seed.

    Returns:
        Dictionary mapping model name to fitted estimator.

    Raises:
        OSError: If a model file cannot be saved.
        ValueError: If an unknown model name appears in config.
    """
    seed = config.get("random_seed", 42)
    if "random_seed" not in config:
        print("Warning: config missing random_seed; using default 42")
    model_names = config.get("models", ["logistic_regression", "linear_svm", "naive_bayes"])
    if "models" not in config:
        print("Warning: config missing models; using default list")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fitted: Dict[str, Any] = {}

    for name in model_names:
        model = instantiate_model(name, seed)
        model.fit(X_train_features, y_train)
        path = ARTIFACTS_DIR / f"{name}.joblib"
        try:
            joblib.dump(model, path)
        except OSError as exc:
            raise OSError(f"Failed to save model {name} to {path}: {exc}") from exc
        fitted[name] = model
        print(f"Trained and saved model: {name} -> {path}")

    return fitted
