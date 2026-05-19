"""Select the best model by configured metric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

ARTIFACTS_DIR = Path("artifacts")

METRIC_KEYS = {
    "macro_f1": "macro_f1",
    "macro_precision": "macro_precision",
    "macro_recall": "macro_recall",
    "accuracy": "accuracy",
}


def _score_for_model(metrics: Dict[str, Any], metric_key: str) -> float:
    """Read a metric value from a model's metrics entry.

    Args:
        metrics: Single-model metrics sub-dictionary.
        metric_key: Key to read (e.g. macro_f1).

    Returns:
        Metric value as float (0.0 if missing).

    Raises:
        None.
    """
    return float(metrics.get(metric_key, 0.0))


def compute_winner(
    metrics_dict: Dict[str, Dict[str, Any]], config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Pick the winning model using selection_metric and tie-break rules.

    Tie-breaking: higher macro_precision, then alphabetically earlier name.

    Args:
        metrics_dict: Per-model metrics from evaluate_models.
        config: Pipeline config with selection_metric.

    Returns:
        Tuple of (winner_name, selection_report dict).

    Raises:
        ValueError: If metrics_dict is empty.
    """
    selection_metric = config.get("selection_metric", "macro_f1")
    if "selection_metric" not in config:
        print("Warning: config missing selection_metric; using macro_f1")
    metric_key = METRIC_KEYS.get(selection_metric, selection_metric)

    if not metrics_dict:
        raise ValueError("metrics_dict is empty; cannot select a winner")

    all_scores = {name: _score_for_model(m, metric_key) for name, m in metrics_dict.items()}
    max_score = max(all_scores.values())
    tied = [name for name, sc in all_scores.items() if sc == max_score]

    tie_broken_by = None
    winner = tied[0]

    if len(tied) > 1:
        precisions = {name: _score_for_model(metrics_dict[name], "macro_precision") for name in tied}
        max_prec = max(precisions.values())
        tied_prec = [name for name in tied if precisions[name] == max_prec]
        if len(tied_prec) > 1:
            tie_broken_by = "alphabetical"
            winner = sorted(tied_prec)[0]
            rationale = (
                f"Tie on {selection_metric}={max_score:.4f} among {tied}; "
                f"resolved alphabetically -> {winner}"
            )
        else:
            tie_broken_by = "macro_precision"
            winner = tied_prec[0]
            rationale = (
                f"Tie on {selection_metric}={max_score:.4f} among {tied}; "
                f"higher macro_precision -> {winner}"
            )
    else:
        rationale = (
            f"Highest {selection_metric}={max_score:.4f} among all models -> {winner}"
        )

    selection_report: Dict[str, Any] = {
        "winner": winner,
        "selection_metric": selection_metric,
        "winner_score": float(all_scores[winner]),
        "all_scores": {k: float(v) for k, v in all_scores.items()},
        "tie_broken_by": tie_broken_by,
        "rationale": rationale,
    }

    return winner, selection_report


def select_winner(
    metrics_dict: Dict[str, Dict[str, Any]], config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Select winner and persist model_selection_report.json.

    Args:
        metrics_dict: Per-model metrics from evaluate_models.
        config: Pipeline config with selection_metric.

    Returns:
        Tuple of (winner_name, selection_report dict).

    Raises:
        OSError: If model_selection_report.json cannot be written.
        ValueError: If metrics_dict is empty.
    """
    winner, selection_report = compute_winner(metrics_dict, config)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "model_selection_report.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(selection_report, fh, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write {path}: {exc}") from exc
    return winner, selection_report


def select_winner_from_file(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Load metrics.json and run select_winner (for validate.py reproducibility).

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple of (winner_name, selection_report).

    Raises:
        OSError: If metrics.json cannot be read.
    """
    path = ARTIFACTS_DIR / "metrics.json"
    try:
        with open(path, encoding="utf-8") as fh:
            metrics_dict = json.load(fh)
    except OSError as exc:
        raise OSError(f"Failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OSError(f"Invalid JSON in {path}: {exc}") from exc
    return compute_winner(metrics_dict, config)
