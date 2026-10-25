"""Baseline comparison and serialized artifact gate for model candidates."""

import json
from pathlib import Path

from .reproducibility import file_sha256


def evaluate_model_gate(evaluation_path, baseline_path, config_path="configs/quality_gates.yaml"):
    import yaml

    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    gate_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    reasons = []

    fresh_baseline = baseline.get("fresh_run", {})
    baseline_accuracy = fresh_baseline.get("overall_accuracy")
    if not fresh_baseline.get("completed") or baseline_accuracy is None:
        reasons.append("Canonical baseline has no completed fresh evaluation; historical metrics cannot authorize promotion")

    report_run = evaluation.get("run", {})
    evaluation_accuracy = evaluation.get("accuracy")
    checkpoint = Path(report_run.get("config", {}).get("checkpoint", ""))
    if not report_run.get("mlflow_run_id"):
        reasons.append("Evaluation report has no MLflow run ID")
    if not report_run.get("git_revision"):
        reasons.append("Evaluation report has no Git revision")
    if report_run.get("git_working_tree_dirty") is not False:
        reasons.append("Evaluation must be produced from a clean committed working tree")
    manifest = report_run.get("dataset_manifest", {})
    if not manifest.get("sha256"):
        reasons.append("Evaluation report has no dataset manifest checksum")
    baseline_manifest_sha = fresh_baseline.get("dataset_manifest_sha256")
    if baseline_manifest_sha and manifest.get("sha256") != baseline_manifest_sha:
        reasons.append("Candidate and canonical baseline do not use the same dataset manifest")
    if not checkpoint.is_file():
        reasons.append(f"Evaluation checkpoint is missing: {checkpoint}")

    size_bytes = checkpoint.stat().st_size if checkpoint.is_file() else None
    artifact_sha256 = file_sha256(checkpoint) if checkpoint.is_file() else None
    max_size = int(gate_config["max_artifact_size_bytes"])
    if size_bytes is not None and size_bytes >= max_size:
        reasons.append(f"Checkpoint size {size_bytes} bytes is not below the strict {max_size}-byte limit")

    if checkpoint.is_file() and report_run.get("checkpoint_sha256") != artifact_sha256:
        reasons.append("Checkpoint SHA-256 differs from the artifact evaluated by this report")
    if baseline_accuracy is not None and evaluation_accuracy is not None:
        minimum = float(baseline_accuracy) + float(gate_config.get("minimum_accuracy_delta", 0.0))
        if float(evaluation_accuracy) < minimum:
            reasons.append(f"Candidate accuracy {evaluation_accuracy:.6f} is below required {minimum:.6f}")
    elif evaluation_accuracy is None:
        reasons.append("Evaluation report has no overall accuracy")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "evaluation": {
            "mlflow_run_id": report_run.get("mlflow_run_id"),
            "git_revision": report_run.get("git_revision"),
            "data_version": report_run.get("data_version"),
            "data_manifest_sha256": manifest.get("sha256"),
            "accuracy": evaluation_accuracy,
        },
        "baseline": {"path": str(baseline_path), "fresh_accuracy": baseline_accuracy},
        "artifact": {
            "path": str(checkpoint),
            "sha256": artifact_sha256,
            "size_bytes": size_bytes,
            "max_size_bytes_exclusive": max_size,
        },
    }
