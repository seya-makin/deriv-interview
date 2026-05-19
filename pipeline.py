#!/usr/bin/env python3
"""Main entry point: run all ML pipeline stages in order."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from stages.artifacts import (
    export_run_manifest,
    generate_html_report,
    generate_test_predictions,
    run_error_analysis,
)
from stages.data_loader import load_and_validate
from stages.evaluator import evaluate_models, extract_feature_importance, run_cross_validation
from stages.features import fit_vectorizer
from stages.preprocessor import preprocess
from stages.safeguards import run_safeguards
from stages.selector import select_winner
from stages.splitter import create_split
from stages.trainer import train_models

ARTIFACTS_DIR = Path("artifacts")
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = ARTIFACTS_DIR / "pipeline.log"


@dataclass
class PipelineState:
    """Shared state passed between pipeline stages."""

    config: Dict[str, Any] = field(default_factory=dict)
    train_df: Any = None
    test_df: Any = None
    validation_report: Dict[str, Any] = field(default_factory=dict)
    preprocess_preview: List[Dict[str, Any]] = field(default_factory=list)
    X_train: Any = None
    X_val: Any = None
    y_train: Any = None
    y_val: Any = None
    val_ids: Any = None
    split_report: Dict[str, Any] = field(default_factory=dict)
    vectorizer: Any = None
    X_train_features: Any = None
    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    winner_name: str = ""
    selection_report: Dict[str, Any] = field(default_factory=dict)
    safeguards_report: Dict[str, Any] = field(default_factory=dict)
    cross_validation_report: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    artifact_paths: List[str] = field(default_factory=list)


def _bootstrap_config() -> Dict[str, Any]:
    """Load config.json before logging is configured.

    Args:
        None.

    Returns:
        Parsed configuration dictionary.

    Raises:
        SystemExit: If config.json is missing, invalid JSON, or unreadable.
    """
    config_path = PROJECT_ROOT / "config.json"
    try:
        with open(config_path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        print(f"config.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in config.json: {exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"Failed to read config.json: {exc}", file=sys.stderr)
        sys.exit(1)


def _configure_logging(config: Dict[str, Any]) -> None:
    """Configure file and stream logging for the pipeline.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        None.

    Raises:
        None. Falls back to stream-only logging if file handler setup fails.
    """
    if not config.get("enable_file_logging", True):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )
        return
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(LOG_PATH),
                logging.StreamHandler(),
            ],
            force=True,
        )
    except Exception as exc:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.warning("File logging disabled: %s", exc)


def _stage(name: str) -> Callable:
    """Create a stage decorator that logs boundaries and records fatal errors.

    Args:
        name: Human-readable stage name for logging.

    Returns:
        Decorator that wraps a stage function accepting PipelineState.

    Raises:
        None.
    """

    def decorator(func: Callable[[PipelineState], PipelineState]) -> Callable:
        def wrapper(state: PipelineState) -> PipelineState:
            logging.info("=== %s (start) ===", name)
            start_time = time.time()
            try:
                result = func(state)
            except Exception as exc:
                logging.error("Stage %s failed: %s", name, exc)
                sys.exit(1)
            elapsed = round(time.time() - start_time, 3)
            try:
                if state.config.get("enable_stage_timings", True):
                    state.stage_timings[name] = elapsed
            except Exception:
                pass
            logging.info("=== %s (complete) ===", name)
            return result

        return wrapper

    return decorator


def _collect_artifact_paths() -> List[str]:
    """List all files currently in artifacts/.

    Args:
        None.

    Returns:
        Sorted list of artifact file path strings.

    Raises:
        None.
    """
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(str(p) for p in ARTIFACTS_DIR.iterdir() if p.is_file())


@_stage("INIT")
def stage_init(state: PipelineState) -> PipelineState:
    """Initialize artifacts directory (config loaded in main before logging).

    Args:
        state: Current pipeline state.

    Returns:
        Updated pipeline state.

    Raises:
        None.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return state


@_stage("DATA_LOADED")
def stage_data_loaded(state: PipelineState) -> PipelineState:
    """Placeholder stage before data validation completes.

    Args:
        state: Current pipeline state.

    Returns:
        Unmodified pipeline state.

    Raises:
        None.
    """
    return state


@_stage("DATA_VALIDATED")
def stage_data_validated(state: PipelineState) -> PipelineState:
    """Load and validate train/test CSV files.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with train_df, test_df, and validation_report.

    Raises:
        None. Underlying loader may call sys.exit on validation failure.
    """
    train_df, test_df, report = load_and_validate(state.config)
    state.train_df = train_df
    state.test_df = test_df
    state.validation_report = report
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("TEXT_PREPROCESSED")
def stage_text_preprocessed(state: PipelineState) -> PipelineState:
    """Preprocess text in train and test sets.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with preprocessed DataFrames and preview.

    Raises:
        None. Preprocessor may raise OSError on artifact write failure.
    """
    train_df, test_df, preview = preprocess(state.train_df, state.test_df, state.config)
    state.train_df = train_df
    state.test_df = test_df
    state.preprocess_preview = preview
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("SPLIT_CREATED")
def stage_split_created(state: PipelineState) -> PipelineState:
    """Create train/validation split and run safeguards.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with split data and safeguards report.

    Raises:
        None. Splitter or safeguards may raise OSError on write failure.
    """
    X_train, X_val, y_train, y_val, val_ids, split_report = create_split(
        state.train_df, state.config
    )
    state.X_train = X_train
    state.X_val = X_val
    state.y_train = y_train
    state.y_val = y_val
    state.val_ids = val_ids
    state.split_report = split_report
    state.safeguards_report = run_safeguards(
        state.train_df, X_train, X_val, y_train, y_val, state.config
    )
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("FEATURES_FIT")
def stage_features_fit(state: PipelineState) -> PipelineState:
    """Fit TF-IDF vectorizer on training split only.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with vectorizer and training features.

    Raises:
        None. Feature stage may raise OSError on save failure.
    """
    vectorizer, X_train_features, _ = fit_vectorizer(state.X_train, state.config)
    state.vectorizer = vectorizer
    state.X_train_features = X_train_features
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("MODELS_TRAINED")
def stage_models_trained(state: PipelineState) -> PipelineState:
    """Train all configured models.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with fitted models dictionary.

    Raises:
        None. Trainer may raise OSError or ValueError on failure.
    """
    state.models = train_models(state.X_train_features, state.y_train, state.config)
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("CROSS_VALIDATION")
def stage_cross_validation(state: PipelineState) -> PipelineState:
    """Run stratified k-fold cross-validation when configured.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with cross_validation_report when enabled.

    Raises:
        None.
    """
    if "cross_validation_folds" not in state.config:
        logging.info("Skipping cross-validation (cross_validation_folds not in config)")
        return state
    state.cross_validation_report = run_cross_validation(
        state.config["models"],
        state.vectorizer,
        state.X_train,
        state.y_train,
        state.config,
    )
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("MODELS_EVALUATED")
def stage_models_evaluated(state: PipelineState) -> PipelineState:
    """Evaluate models on the validation set.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with per-model validation metrics.

    Raises:
        None. Evaluator may raise OSError on metrics write failure.
    """
    state.metrics = evaluate_models(
        state.models,
        state.vectorizer,
        state.X_val,
        state.y_val,
        state.config,
    )
    state.artifact_paths = _collect_artifact_paths()
    return state


def _run_feature_importance(state: PipelineState) -> PipelineState:
    """Run optional feature importance extraction without failing the pipeline.

    Args:
        state: Current pipeline state.

    Returns:
        Unmodified pipeline state.

    Raises:
        None.
    """
    if not state.config.get("feature_importance", True):
        logging.info("Skipping feature importance (disabled in config)")
        return state
    logging.info("=== FEATURE_IMPORTANCE (start) ===")
    try:
        label_names = sorted(state.y_train.unique(), key=str)
        extract_feature_importance(
            state.models,
            state.vectorizer,
            [str(l) for l in label_names],
            state.config,
        )
    except Exception as exc:
        logging.warning("Feature importance stage failed: %s", exc)
    logging.info("=== FEATURE_IMPORTANCE (complete) ===")
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("WINNER_SELECTED")
def stage_winner_selected(state: PipelineState) -> PipelineState:
    """Select the best model by configured metric.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with winner_name and selection_report.

    Raises:
        None. Selector may raise OSError or ValueError on failure.
    """
    winner, report = select_winner(state.metrics, state.config)
    state.winner_name = winner
    state.selection_report = report
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("ARTIFACTS_SAVED")
def stage_artifacts_saved(state: PipelineState) -> PipelineState:
    """Run error analysis for the winning model.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with error analysis artifact written.

    Raises:
        None. Error analysis may raise OSError on write failure.
    """
    run_error_analysis(
        state.winner_name,
        state.models,
        state.vectorizer,
        state.X_val,
        state.y_val,
        state.val_ids,
        state.X_val,
        state.config,
    )
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("TEST_PREDICTIONS_GENERATED")
def stage_test_predictions(state: PipelineState) -> PipelineState:
    """Generate test set predictions from saved artifacts.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state after test predictions are saved.

    Raises:
        None. May raise OSError or FileNotFoundError from artifact loading.
    """
    generate_test_predictions(
        state.winner_name,
        state.models,
        state.vectorizer,
        state.test_df,
        state.config,
    )
    state.artifact_paths = _collect_artifact_paths()
    return state


@_stage("REPORT_EXPORTED")
def stage_report_exported(state: PipelineState) -> PipelineState:
    """Export run manifest and optional HTML summary report.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with final artifact paths.

    Raises:
        None. Manifest export may raise OSError; HTML report failures are logged.
    """
    state.artifact_paths = _collect_artifact_paths()
    export_run_manifest(
        state.config,
        state.metrics,
        state.winner_name,
        state.artifact_paths,
        stage_timings=state.stage_timings if state.stage_timings else None,
    )
    try:
        generate_html_report(state.config)
    except Exception as exc:
        logging.warning("HTML report generation failed: %s", exc)
    state.artifact_paths = _collect_artifact_paths()
    return state


STAGES: List[Tuple[str, Callable[[PipelineState], PipelineState]]] = [
    ("INIT", stage_init),
    ("DATA_LOADED", stage_data_loaded),
    ("DATA_VALIDATED", stage_data_validated),
    ("TEXT_PREPROCESSED", stage_text_preprocessed),
    ("SPLIT_CREATED", stage_split_created),
    ("FEATURES_FIT", stage_features_fit),
    ("MODELS_TRAINED", stage_models_trained),
    ("CROSS_VALIDATION", stage_cross_validation),
    ("MODELS_EVALUATED", stage_models_evaluated),
    ("WINNER_SELECTED", stage_winner_selected),
    ("ARTIFACTS_SAVED", stage_artifacts_saved),
    ("TEST_PREDICTIONS_GENERATED", stage_test_predictions),
    ("REPORT_EXPORTED", stage_report_exported),
]


def main() -> None:
    """Run the full pipeline and log an artifact summary.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If a required stage fails.
    """
    os.chdir(PROJECT_ROOT)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap_config = _bootstrap_config()
    _configure_logging(bootstrap_config)
    state = PipelineState(config=bootstrap_config)
    for idx, (name, stage_fn) in enumerate(STAGES):
        state = stage_fn(state)
        if name == "MODELS_EVALUATED":
            fi_start = time.time()
            state = _run_feature_importance(state)
            try:
                if state.config.get("enable_stage_timings", True):
                    state.stage_timings["FEATURE_IMPORTANCE"] = round(
                        time.time() - fi_start, 3
                    )
            except Exception:
                pass

    logging.info("=== Pipeline complete ===")
    logging.info("Winning model: %s", state.winner_name)
    logging.info("Artifact paths:")
    for path in state.artifact_paths:
        logging.info("  - %s", path)


if __name__ == "__main__":
    main()
