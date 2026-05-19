"""Error analysis, test predictions, and run manifest export."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from stages.features import transform
from stages.preprocessor import preprocess_text

ARTIFACTS_DIR = Path("artifacts")


def _score_sample(model, features_row) -> Optional[float]:
    """Confidence or decision score for a single sample.

    Args:
        model: Fitted sklearn classifier.
        features_row: Sparse matrix with a single row.

    Returns:
        Confidence or absolute decision score, or None if unsupported.

    Raises:
        None.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_row)
        return float(np.max(proba))
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features_row)
        if np.ndim(scores) == 0:
            return float(abs(scores))
        return float(np.max(np.abs(scores)))
    return None


def run_error_analysis(
    winner_name: str,
    models: Dict[str, Any],
    vectorizer,
    X_val: pd.Series,
    y_val: pd.Series,
    val_ids: pd.Series,
    val_texts: pd.Series,
    config: Dict[str, Any],
) -> None:
    """Analyze misclassified validation examples for the winning model.

    Args:
        winner_name: Name of the selected model.
        models: All fitted models (uses winner from this dict).
        vectorizer: Fitted vectorizer.
        X_val: Validation texts (preprocessed).
        y_val: True validation labels.
        val_ids: Validation sample ids aligned with X_val.
        val_texts: Raw or processed validation texts for reporting.
        config: Pipeline config with top_k_error_examples.

    Raises:
        OSError: If error_analysis.json cannot be written.
        KeyError: If winner_name is not in models.
    """
    top_k = config.get("top_k_error_examples", 10)
    if "top_k_error_examples" not in config:
        print("Warning: config missing top_k_error_examples; using 10")

    model = models[winner_name]
    X_features = transform(vectorizer, X_val)
    y_pred = model.predict(X_features)

    errors: List[Dict[str, Any]] = []
    for i in range(len(y_val)):
        true_lbl = y_val.iloc[i]
        pred_lbl = y_pred[i]
        if true_lbl != pred_lbl:
            row_features = X_features[i : i + 1]
            score = _score_sample(model, row_features)
            errors.append(
                {
                    "id": val_ids.iloc[i],
                    "text": val_texts.iloc[i],
                    "true_label": str(true_lbl),
                    "predicted_label": str(pred_lbl),
                    "confidence_or_score": score,
                    "_sort_score": score if score is not None else 0.0,
                }
            )

    errors.sort(key=lambda x: x["_sort_score"], reverse=True)
    k = min(top_k, len(errors))
    output: List[Dict[str, Any]] = []
    for item in errors[:k]:
        score = item["confidence_or_score"]
        score_str = f"{score:.4f}" if score is not None else "null"
        reason = (
            f"Predicted {item['predicted_label']} but true label is "
            f"{item['true_label']} with score {score_str}"
        )
        output.append(
            {
                "id": str(item["id"]),
                "text": str(item["text"]),
                "true_label": item["true_label"],
                "predicted_label": item["predicted_label"],
                "confidence_or_score": score,
                "reason": reason,
            }
        )

    path = ARTIFACTS_DIR / "error_analysis.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write {path}: {exc}") from exc


def generate_test_predictions(
    winner_name: str,
    models: Dict[str, Any],
    vectorizer,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Load artifacts from disk, preprocess test texts, predict, save CSV.

    Args:
        winner_name: Selected model name (used to locate joblib file).
        models: Unused at inference; models are loaded from disk.
        vectorizer: Unused; vectorizer loaded from disk.
        test_df: Test DataFrame with id and text columns.
        config: Pipeline configuration (unused).

    Raises:
        OSError: If artifact files cannot be loaded or predictions cannot be saved.
        FileNotFoundError: If required joblib files are missing.
    """
    _ = models, vectorizer, config
    vec_path = ARTIFACTS_DIR / "vectorizer.joblib"
    model_path = ARTIFACTS_DIR / f"{winner_name}.joblib"

    try:
        loaded_vectorizer = joblib.load(vec_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Vectorizer not found at {vec_path}. Run pipeline.py first."
        ) from exc
    except OSError as exc:
        raise OSError(f"Failed to load vectorizer from {vec_path}: {exc}") from exc

    try:
        loaded_model = joblib.load(model_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run pipeline.py first."
        ) from exc
    except OSError as exc:
        raise OSError(f"Failed to load model from {model_path}: {exc}") from exc

    # Read test.csv from disk so row count matches the source file (includes empty-text rows).
    project_root = Path(__file__).resolve().parent.parent
    test_path = project_root / "test.csv"
    try:
        predict_df = pd.read_csv(test_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise OSError(f"Failed to read {test_path} for test predictions: {exc}") from exc
    _ = test_df

    ids = []
    predictions = []
    for _, row in predict_df.iterrows():
        raw_text = row["text"]
        if raw_text is None or (isinstance(raw_text, float) and np.isnan(raw_text)):
            processed = ""
        else:
            processed = preprocess_text(raw_text)

        if not processed.strip():
            logging.warning(
                "Empty text after preprocessing for test id %s; predicted_label=unknown",
                row["id"],
            )
            ids.append(row["id"])
            predictions.append("unknown")
            continue

        features = transform(loaded_vectorizer, [processed])
        pred = loaded_model.predict(features)[0]
        ids.append(row["id"])
        predictions.append(pred)

    out_df = pd.DataFrame({"id": ids, "predicted_label": predictions})
    path = ARTIFACTS_DIR / "test_predictions.csv"
    try:
        out_df.to_csv(path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {path}: {exc}") from exc
    print(f"Test predictions saved to {path}")


def export_run_manifest(
    config: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    winner: str,
    artifact_paths: List[str],
    stage_timings: Optional[Dict[str, float]] = None,
) -> None:
    """Write run_manifest.json summarizing the pipeline run.

    Args:
        config: Pipeline configuration dictionary.
        metrics: Full metrics dictionary keyed by model name.
        winner: Winning model name.
        artifact_paths: List of artifact file paths produced.
        stage_timings: Optional mapping of stage name to elapsed seconds.

    Returns:
        None.

    Raises:
        OSError: If run_manifest.json cannot be written.
    """
    winner_metrics = metrics.get(winner, {})
    paths = list(artifact_paths)
    log_path = str(ARTIFACTS_DIR / "pipeline.log")
    if Path(log_path).exists() and log_path not in paths:
        paths.append(log_path)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.get("random_seed"),
        "files_read": ["config.json", "train.csv", "test.csv"],
        "models_trained": config.get("models", []),
        "winning_model": winner,
        "key_metrics": {
            "accuracy": winner_metrics.get("accuracy"),
            "macro_precision": winner_metrics.get("macro_precision"),
            "macro_recall": winner_metrics.get("macro_recall"),
            "macro_f1": winner_metrics.get("macro_f1"),
        },
        "output_artifact_paths": sorted(paths),
    }
    if stage_timings:
        manifest["stage_timings_seconds"] = stage_timings
    path = ARTIFACTS_DIR / "run_manifest.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write {path}: {exc}") from exc


def generate_html_report(config: Dict[str, Any]) -> None:
    """Generate a human-readable HTML summary of the pipeline run.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        None. Saves ``artifacts/report.html`` when successful.

    Raises:
        None. Logs a warning and skips if any artifact is missing or malformed.
    """
    if not config.get("html_report", True):
        return
    try:
        metrics = _read_json_artifact("metrics.json")
        selection = _read_json_artifact("model_selection_report.json")
        safeguards = _read_json_artifact("safeguards_report.json")
        cv_path = ARTIFACTS_DIR / "cross_validation_report.json"
        cv_report = None
        if cv_path.exists():
            cv_report = _read_json_artifact("cross_validation_report.json")

        winner = selection.get("winner", "unknown")
        winner_score = selection.get("winner_score", 0.0)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        rows_html = []
        for model_name, model_metrics in metrics.items():
            is_winner = model_name == winner
            row_style = ' style="background-color: #e8f5e9;"' if is_winner else ""
            rows_html.append(
                f"<tr{row_style}>"
                f"<td>{model_name}</td>"
                f"<td>{model_metrics.get('accuracy', 0):.4f}</td>"
                f"<td>{model_metrics.get('macro_precision', 0):.4f}</td>"
                f"<td>{model_metrics.get('macro_recall', 0):.4f}</td>"
                f"<td>{model_metrics.get('macro_f1', 0):.4f}</td>"
                f"</tr>"
            )

        safeguards_list = "".join(
            f"<li>{w}</li>" for w in safeguards.get("warnings", [])
        )
        if not safeguards_list:
            safeguards_list = "<li>No warnings</li>"

        cv_section = ""
        if cv_report:
            cv_rows = []
            for model_name, scores in cv_report.items():
                cv_rows.append(
                    f"<tr><td>{model_name}</td>"
                    f"<td>{scores.get('mean', 0):.4f}</td>"
                    f"<td>{scores.get('std', 0):.4f}</td></tr>"
                )
            cv_section = f"""
            <h2>Cross-Validation (macro F1)</h2>
            <table>
              <tr><th>Model</th><th>Mean</th><th>Std</th></tr>
              {''.join(cv_rows)}
            </table>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pipeline Run Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
    h1 {{ color: #1a5276; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
    ul {{ line-height: 1.6; }}
  </style>
</head>
<body>
  <h1>Pipeline Run Report</h1>
  <p><strong>Timestamp:</strong> {timestamp}</p>
  <h2>Winner</h2>
  <p><strong>{winner}</strong> — {selection.get('selection_metric', 'macro_f1')} =
     {winner_score:.4f}</p>
  <h2>Metrics</h2>
  <table>
    <tr>
      <th>Model</th><th>Accuracy</th><th>Macro Precision</th>
      <th>Macro Recall</th><th>Macro F1</th>
    </tr>
    {''.join(rows_html)}
  </table>
  <h2>Safeguards</h2>
  <ul>{safeguards_list}</ul>
  {cv_section}
</body>
</html>
"""
        out_path = ARTIFACTS_DIR / "report.html"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
    except Exception as exc:
        print(f"Warning: HTML report generation failed: {exc}")


def _read_json_artifact(filename: str) -> Dict[str, Any]:
    """Load a JSON artifact from the artifacts directory.

    Args:
        filename: Base name of the JSON file under artifacts/.

    Returns:
        Parsed JSON object as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: If the file cannot be read.
    """
    path = ARTIFACTS_DIR / filename
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
