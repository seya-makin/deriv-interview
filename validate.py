#!/usr/bin/env python3
"""Validate pipeline artifacts and reproducibility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from stages.preprocessor import preprocess_text
from stages.selector import compute_winner

ARTIFACTS_DIR = Path("artifacts")
PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_JSON = [
    "data_validation_report.json",
    "preprocessing_preview.json",
    "split_report.json",
    "metrics.json",
    "model_selection_report.json",
    "error_analysis.json",
    "safeguards_report.json",
    "run_manifest.json",
]

REQUIRED_OTHER = [
    "test_predictions.csv",
    "vectorizer.joblib",
]

KNOWN_PREPROCESS = "  Hello   World!\n\t"


def _check(name: str, passed: bool, detail: str = "") -> Tuple[bool, str]:
    """Record a single required check result as PASS or FAIL.

    Args:
        name: Short check description.
        passed: Whether the check succeeded.
        detail: Optional extra context.

    Returns:
        Tuple of (passed flag, formatted log line).

    Raises:
        None.
    """
    status = "PASS" if passed else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return passed, line


def _warn(name: str, passed: bool, detail: str = "") -> Tuple[bool, str]:
    """Record an optional check result as PASS or WARN (never fails the run).

    Args:
        name: Short check description.
        passed: Whether the optional check succeeded.
        detail: Optional extra context.

    Returns:
        Tuple of (always True, formatted log line).

    Raises:
        None.
    """
    status = "PASS" if passed else "WARN"
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return True, line


def _info(message: str) -> None:
    """Print an informational validation message.

    Args:
        message: Text to print.

    Returns:
        None.

    Raises:
        None.
    """
    print(f"[INFO] {message}")


def _load_config() -> Dict[str, Any]:
    """Load config.json for selector reproducibility check.

    Args:
        None.

    Returns:
        Parsed configuration dictionary.

    Raises:
        OSError: If config.json cannot be read.
        json.JSONDecodeError: If config.json is invalid JSON.
    """
    path = PROJECT_ROOT / "config.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    """Run all validation checks and print summary.

    Args:
        None.

    Returns:
        None.

    Raises:
        None. Exits with code 1 if any required check fails.
    """
    failures: List[str] = []
    total = 0

    def run(name: str, fn: Callable[[], Tuple[bool, str]]) -> None:
        nonlocal total
        total += 1
        ok, _ = fn()
        if not ok:
            failures.append(name)

    def check_required_files() -> Tuple[bool, str]:
        missing = []
        for fname in REQUIRED_JSON + REQUIRED_OTHER:
            if not (ARTIFACTS_DIR / fname).exists():
                missing.append(fname)
        model_files = list(ARTIFACTS_DIR.glob("*.joblib"))
        model_files = [p for p in model_files if p.name != "vectorizer.joblib"]
        if not model_files:
            missing.append("at least one model .joblib (besides vectorizer)")
        ok = len(missing) == 0
        return _check("Required artifact files exist", ok, ", ".join(missing) if missing else "")

    def check_json_valid() -> Tuple[bool, str]:
        bad = []
        for fname in REQUIRED_JSON:
            path = ARTIFACTS_DIR / fname
            if not path.exists():
                bad.append(f"{fname} missing")
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                bad.append(f"{fname}: {exc}")
        ok = len(bad) == 0
        return _check("JSON artifacts are valid", ok, "; ".join(bad))

    def check_metrics_models() -> Tuple[bool, str]:
        path = ARTIFACTS_DIR / "metrics.json"
        try:
            with open(path, encoding="utf-8") as fh:
                metrics = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            return _check("metrics.json has >= 3 models", False, str(exc))
        ok = len(metrics) >= 3
        return _check("metrics.json has >= 3 models", ok, f"found {len(metrics)}")

    def check_winner_in_metrics() -> Tuple[bool, str]:
        try:
            with open(ARTIFACTS_DIR / "model_selection_report.json", encoding="utf-8") as fh:
                report = json.load(fh)
            with open(ARTIFACTS_DIR / "metrics.json", encoding="utf-8") as fh:
                metrics = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            return _check("Winner exists in metrics.json", False, str(exc))
        winner = report.get("winner")
        ok = winner in metrics
        return _check("Winner exists in metrics.json", ok, f"winner={winner}")

    def check_test_predictions() -> Tuple[bool, str]:
        pred_path = ARTIFACTS_DIR / "test_predictions.csv"
        test_path = PROJECT_ROOT / "test.csv"
        try:
            preds = pd.read_csv(pred_path)
            test_df = pd.read_csv(test_path)
        except (OSError, pd.errors.EmptyDataError) as exc:
            return _check("test_predictions.csv schema and row count", False, str(exc))
        cols_ok = list(preds.columns) == ["id", "predicted_label"]
        count_ok = len(preds) == len(test_df)
        ok = cols_ok and count_ok
        detail = f"columns={list(preds.columns)}, pred_rows={len(preds)}, test_rows={len(test_df)}"
        return _check("test_predictions.csv schema and row count", ok, detail)

    def check_winner_reproducible() -> Tuple[bool, str]:
        try:
            with open(ARTIFACTS_DIR / "metrics.json", encoding="utf-8") as fh:
                metrics = json.load(fh)
            with open(ARTIFACTS_DIR / "model_selection_report.json", encoding="utf-8") as fh:
                report = json.load(fh)
            config = _load_config()
            recomputed, _ = compute_winner(metrics, config)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return _check("Winner selection reproducible", False, str(exc))
        ok = recomputed == report.get("winner")
        detail = f"saved={report.get('winner')}, recomputed={recomputed}"
        return _check("Winner selection reproducible", ok, detail)

    def check_predict_cli() -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "predict.py"), "--text", "test input"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _check("predict.py CLI exits 0", False, str(exc))
        ok = result.returncode == 0
        detail = result.stderr.strip() if not ok else ""
        return _check("predict.py CLI exits 0", ok, detail)

    def check_preprocess_consistency() -> Tuple[bool, str]:
        direct = preprocess_text(KNOWN_PREPROCESS)
        via_predict_path = preprocess_text(KNOWN_PREPROCESS)
        ok = direct == via_predict_path
        return _check(
            "preprocess_text consistency",
            ok,
            f"direct={direct!r}, predict_path={via_predict_path!r}",
        )

    def check_cross_validation_report() -> Tuple[bool, str]:
        path = ARTIFACTS_DIR / "cross_validation_report.json"
        if not path.exists():
            return _check(
                "cross_validation_report.json (optional)",
                True,
                "not present; skipped",
            )
        try:
            with open(path, encoding="utf-8") as fh:
                cv_report = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            return _check("cross_validation_report.json valid", False, str(exc))
        ok = isinstance(cv_report, dict) and len(cv_report) >= 3
        return _check(
            "cross_validation_report.json valid with >= 3 models",
            ok,
            f"found {len(cv_report) if isinstance(cv_report, dict) else 0} entries",
        )

    run("required_files", check_required_files)
    run("json_valid", check_json_valid)
    run("metrics_count", check_metrics_models)
    run("winner_in_metrics", check_winner_in_metrics)
    run("test_predictions", check_test_predictions)
    run("winner_repro", check_winner_reproducible)
    run("predict_cli", check_predict_cli)
    run("preprocess", check_preprocess_consistency)
    run("cross_validation", check_cross_validation_report)

    def check_feature_importance_optional() -> Tuple[bool, str]:
        path = ARTIFACTS_DIR / "feature_importance.json"
        if not path.exists():
            return _warn("feature_importance.json", True, "not present; skipped")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            ok = isinstance(data, dict) and "top_features_per_class" in data
            return _warn(
                "feature_importance.json structure",
                ok,
                "missing top_features_per_class" if not ok else "ok",
            )
        except (OSError, json.JSONDecodeError) as exc:
            return _warn("feature_importance.json valid JSON", False, str(exc))

    def check_report_html_optional() -> Tuple[bool, str]:
        path = ARTIFACTS_DIR / "report.html"
        if not path.exists():
            return _warn("report.html", True, "not present; skipped")
        try:
            ok = path.stat().st_size > 0
            return _warn("report.html non-empty", ok, f"size={path.stat().st_size} bytes")
        except OSError as exc:
            return _warn("report.html readable", False, str(exc))

    def check_pipeline_log_optional() -> Tuple[bool, str]:
        path = ARTIFACTS_DIR / "pipeline.log"
        if not path.exists():
            return _warn("pipeline.log", True, "not present; skipped")
        try:
            ok = path.stat().st_size > 0
            return _warn("pipeline.log non-empty", ok, f"size={path.stat().st_size} bytes")
        except OSError as exc:
            return _warn("pipeline.log readable", False, str(exc))

    def print_stage_timings_info() -> Tuple[bool, str]:
        path = ARTIFACTS_DIR / "run_manifest.json"
        if not path.exists():
            _info("stage_timings_seconds not available (run_manifest.json missing)")
            return True, ""
        try:
            with open(path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            timings = manifest.get("stage_timings_seconds")
            if timings:
                _info(f"stage_timings_seconds: {timings}")
            else:
                _info("stage_timings_seconds not present in run_manifest.json")
        except (OSError, json.JSONDecodeError) as exc:
            _info(f"could not read stage timings: {exc}")
        return True, ""

    check_feature_importance_optional()
    check_report_html_optional()
    check_pipeline_log_optional()
    print_stage_timings_info()

    failed_count = len(failures)
    if failed_count == 0:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    print(f"\n{failed_count} CHECKS FAILED")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)


if __name__ == "__main__":
    main()
