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
    if not report_run.get("source_snapshot_sha256"):
        reasons.append("Evaluation is missing its exact source snapshot fingerprint")
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

    # ── Multi-Metric Quality Gates (Stage 4a) ───────────────────────────────
    min_macro_f1 = gate_config.get("min_macro_f1")
    candidate_macro_f1 = evaluation.get("macro_f1")
    if candidate_macro_f1 is None and "confusion_matrix" in evaluation:
        cm = evaluation["confusion_matrix"]
        num_classes = len(cm)
        f1s = []
        for i in range(num_classes):
            sup = sum(cm[i][c] for c in range(num_classes))
            pred_sup = sum(cm[r][i] for r in range(num_classes))
            cor = cm[i][i]
            rec = cor / sup if sup else 0.0
            prec = cor / pred_sup if pred_sup else 0.0
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            f1s.append(f1)
        if f1s:
            candidate_macro_f1 = float(sum(f1s) / len(f1s))

    if min_macro_f1 is not None:
        if candidate_macro_f1 is None:
            reasons.append("Candidate evaluation report has no Macro-F1 score")
        elif float(candidate_macro_f1) < float(min_macro_f1):
            reasons.append(
                f"Candidate Macro-F1 {candidate_macro_f1:.4f} is below required threshold {float(min_macro_f1):.4f}"
            )

    max_latency = gate_config.get("max_mean_latency_ms")
    candidate_latency = evaluation.get("mean_latency_ms")
    if max_latency is not None and candidate_latency is not None:
        if float(candidate_latency) > float(max_latency):
            reasons.append(
                f"Candidate mean latency {candidate_latency:.2f} ms exceeds latency SLA of {float(max_latency):.2f} ms"
            )

    per_class_floors = gate_config.get("min_per_class_recall", {})
    per_class_eval = evaluation.get("per_class", {})
    for class_key, min_recall in per_class_floors.items():
        class_stats = per_class_eval.get(str(class_key), {})
        class_recall = class_stats.get("recall", class_stats.get("accuracy"))
        if class_recall is not None and float(class_recall) < float(min_recall):
            reasons.append(
                f"Class {class_key} recall {class_recall:.4f} is below required floor {float(min_recall):.4f}"
            )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "evaluation": {
            "mlflow_run_id": report_run.get("mlflow_run_id"),
            "git_revision": report_run.get("git_revision"),
            "source_snapshot_sha256": report_run.get("source_snapshot_sha256"),
            "data_version": report_run.get("data_version"),
            "data_manifest_sha256": manifest.get("sha256"),
            "accuracy": evaluation_accuracy,
            "macro_f1": candidate_macro_f1,
            "mean_latency_ms": candidate_latency,
        },
        "baseline": {"path": str(baseline_path), "fresh_accuracy": baseline_accuracy},
        "thresholds": {
            "minimum_accuracy_delta": gate_config.get("minimum_accuracy_delta"),
            "min_macro_f1": min_macro_f1,
            "max_mean_latency_ms": max_latency,
            "min_per_class_recall": per_class_floors,
        },
        "artifact": {
            "path": str(checkpoint),
            "sha256": artifact_sha256,
            "size_bytes": size_bytes,
            "max_size_bytes_exclusive": max_size,
        },
    }
