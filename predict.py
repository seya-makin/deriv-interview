#!/usr/bin/env python3
"""Standalone inference CLI for single text or batch CSV input."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from stages.features import transform
from stages.preprocessor import preprocess_text

DEFAULT_MODEL_DIR = Path("artifacts")
DEFAULT_BATCH_OUTPUT = "batch_predictions.csv"


def _load_winner_name(model_dir: Path) -> str:
    """Read winning model name from model_selection_report.json.

    Args:
        model_dir: Directory containing pipeline artifacts.

    Returns:
        Winner model name string.

    Raises:
        FileNotFoundError: If the selection report is missing.
        OSError: If the file cannot be read or parsed.
    """
    report_path = model_dir / "model_selection_report.json"
    try:
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Selection report not found at {report_path}. "
            "Run pipeline.py first to generate artifacts."
        ) from exc
    except OSError as exc:
        raise OSError(f"Failed to read {report_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OSError(f"Invalid JSON in {report_path}: {exc}") from exc
    return str(report["winner"])


def _load_artifacts(model_dir: Path, winner_name: str) -> Tuple[Any, Any]:
    """Load vectorizer and winning model from model_dir.

    Args:
        model_dir: Directory containing pipeline artifacts.
        winner_name: Name of the winning model file (without extension).

    Returns:
        Tuple of (vectorizer, fitted model).

    Raises:
        FileNotFoundError: If required joblib files are missing.
        OSError: If artifacts cannot be loaded.
    """
    vec_path = model_dir / "vectorizer.joblib"
    model_path = model_dir / f"{winner_name}.joblib"
    try:
        vectorizer = joblib.load(vec_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Vectorizer not found at {vec_path}. Run pipeline.py first to generate artifacts."
        ) from exc
    except OSError as exc:
        raise OSError(f"Failed to load vectorizer: {exc}") from exc

    try:
        model = joblib.load(model_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run pipeline.py first to generate artifacts."
        ) from exc
    except OSError as exc:
        raise OSError(f"Failed to load model: {exc}") from exc

    return vectorizer, model


def _score_from_model(model, features) -> Optional[float]:
    """Compute confidence or decision score for transformed features.

    Args:
        model: Fitted sklearn classifier.
        features: Sparse feature matrix for one or more samples.

    Returns:
        Maximum probability or absolute decision score, or None if unsupported.

    Raises:
        None.
    """
    if hasattr(model, "predict_proba"):
        return float(np.max(model.predict_proba(features)))
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        if np.ndim(raw) == 0:
            return float(abs(raw))
        return float(np.max(np.abs(raw)))
    return None


def _predict_with_score(
    text: str, vectorizer, model
) -> Tuple[str, Optional[float]]:
    """Preprocess, transform, predict label and confidence/score.

    Args:
        text: Raw input text.
        vectorizer: Fitted TfidfVectorizer.
        model: Fitted classifier.

    Returns:
        Tuple of (predicted_label, confidence_or_score).

    Raises:
        None.
    """
    processed = preprocess_text(text)
    if not processed.strip():
        print(
            f"Warning: empty text after preprocessing; label set to unknown",
            file=sys.stderr,
        )
        return "unknown", None
    features = transform(vectorizer, [processed])
    label = str(model.predict(features)[0])
    return label, _score_from_model(model, features)


def _validate_single_input(text: Optional[str]) -> str:
    """Validate CLI text input and return usable string.

    Args:
        text: Raw --text argument value.

    Returns:
        Validated text string.

    Raises:
        SystemExit: If text is None, non-string, or whitespace-only.
    """
    if text is None:
        print("Error: --text is required and cannot be None", file=sys.stderr)
        sys.exit(1)
    if not isinstance(text, str):
        print(f"Error: --text must be a string, got {type(text).__name__}", file=sys.stderr)
        sys.exit(1)
    if not text.strip():
        print("Error: --text cannot be empty or whitespace-only", file=sys.stderr)
        sys.exit(1)
    if len(text) > 10000:
        print(
            f"Warning: input length {len(text)} exceeds 10000 characters; proceeding anyway",
            file=sys.stderr,
        )
    return text


def _run_single_inference(text: str, model_dir: Path) -> None:
    """Run inference for one text string and print formatted output.

    Args:
        text: Input text to classify.
        model_dir: Directory containing trained artifacts.

    Returns:
        None.

    Raises:
        SystemExit: On artifact load or prediction failure.
    """
    try:
        winner_name = _load_winner_name(model_dir)
        vectorizer, model = _load_artifacts(model_dir, winner_name)
        label, score = _predict_with_score(text, vectorizer, model)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    score_display = f"{score:.2f}" if score is not None else "N/A"
    print(f"Predicted label : {label}")
    print(f"Confidence/Score: {score_display}")
    print(f"Model used      : {winner_name}")


def _run_batch_inference(input_file: Path, output_file: Path, model_dir: Path) -> None:
    """Run batch inference on a CSV file and write predictions.

    Args:
        input_file: CSV path with id and text columns.
        output_file: CSV path for predictions output.
        model_dir: Directory containing trained artifacts.

    Returns:
        None.

    Raises:
        SystemExit: On validation, load, or write errors.
    """
    if not input_file.exists():
        print(f"Error: input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Error: failed to read input CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    for col in ("id", "text"):
        if col not in df.columns:
            print(
                f"Error: input CSV must have columns id and text; found {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        winner_name = _load_winner_name(model_dir)
        vectorizer, model = _load_artifacts(model_dir, winner_name)
    except (FileNotFoundError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    try:
        for _, row in df.iterrows():
            raw_text = row["text"]
            display_text = raw_text
            if raw_text is None or (isinstance(raw_text, float) and np.isnan(raw_text)):
                processed = ""
                display_text = ""
            elif not isinstance(raw_text, str):
                display_text = str(raw_text)
                processed = preprocess_text(display_text)
            else:
                processed = preprocess_text(raw_text)

            if not processed.strip():
                print(
                    f"Warning: empty text for id {row['id']}; "
                    "predicted_label=unknown",
                    file=sys.stderr,
                )
                label = "unknown"
                score = None
            else:
                features = transform(vectorizer, [processed])
                label = str(model.predict(features)[0])
                score = _score_from_model(model, features)

            rows.append(
                {
                    "id": row["id"],
                    "text": display_text if isinstance(display_text, str) else "",
                    "predicted_label": label,
                    "confidence_or_score": score,
                }
            )
    except Exception as exc:
        print(f"Error during batch prediction: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_file, index=False)
        print(f"Batch predictions written to {output_file} ({len(out_df)} rows)")
    except OSError as exc:
        print(f"Error: failed to write output CSV: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Parse arguments and run single or batch inference.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: On invalid arguments or inference failure.
    """
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    parser = argparse.ArgumentParser(description="Predict labels for text input.")
    parser.add_argument("--text", help="Single input text to classify")
    parser.add_argument("--input-file", help="CSV with id,text columns for batch inference")
    parser.add_argument(
        "--output-file",
        default=DEFAULT_BATCH_OUTPUT,
        help="Output CSV for batch mode (default: batch_predictions.csv)",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing trained artifacts (default: artifacts/)",
    )
    args = parser.parse_args()

    if not args.text and not args.input_file:
        parser.print_usage()
        print(
            "\nError: provide --text for single inference or --input-file for batch mode.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.text and args.input_file:
        print("Error: use either --text or --input-file, not both.", file=sys.stderr)
        sys.exit(1)

    model_dir = Path(args.model_dir)

    if args.text:
        text = _validate_single_input(args.text)
        _run_single_inference(text, model_dir)
    else:
        _run_batch_inference(Path(args.input_file), Path(args.output_file), model_dir)


if __name__ == "__main__":
    main()
