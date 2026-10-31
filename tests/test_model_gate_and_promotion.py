import json
from pathlib import Path
import yaml

from vehicle_classifier_core.model_gate import evaluate_model_gate


def test_model_gate_evaluates_multi_metric_thresholds(tmp_path):
    # Create mock baseline and evaluation
    baseline_path = tmp_path / "baseline.json"
    eval_path = tmp_path / "evaluation.json"
    config_path = tmp_path / "quality_gates.yaml"
    dummy_checkpoint = tmp_path / "model.pth"
    dummy_checkpoint.write_bytes(b"dummy model bytes")

    from vehicle_classifier_core.reproducibility import file_sha256
    model_sha = file_sha256(dummy_checkpoint)

    baseline_data = {
        "fresh_run": {
            "completed": True,
            "overall_accuracy": 0.90,
            "dataset_manifest_sha256": "manifest-123",
        }
    }
    baseline_path.write_text(json.dumps(baseline_data), encoding="utf-8")

    config_data = {
        "max_artifact_size_bytes": 5000000,
        "minimum_accuracy_delta": -0.01,
        "min_macro_f1": 0.85,
        "max_mean_latency_ms": 15.0,
        "min_per_class_recall": {
            "0": 0.80,
            "1": 0.85,
        },
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    # 1. Compliant candidate
    eval_data = {
        "accuracy": 0.92,
        "macro_f1": 0.91,
        "mean_latency_ms": 10.5,
        "per_class": {
            "0": {"support": 100, "correct": 85, "recall": 0.85},
            "1": {"support": 100, "correct": 90, "recall": 0.90},
        },
        "run": {
            "mlflow_run_id": "run-abc",
            "git_revision": "git-abc",
            "source_snapshot_sha256": "snap-abc",
            "checkpoint_sha256": model_sha,
            "dataset_manifest": {"sha256": "manifest-123"},
            "config": {"checkpoint": str(dummy_checkpoint)},
        },
    }
    eval_path.write_text(json.dumps(eval_data), encoding="utf-8")

    result = evaluate_model_gate(str(eval_path), str(baseline_path), str(config_path))
    assert result["passed"] is True
    assert result["reasons"] == []
    assert result["evaluation"]["macro_f1"] == 0.91
    assert result["evaluation"]["mean_latency_ms"] == 10.5

    # 2. Reject on Macro-F1 violation
    eval_data["macro_f1"] = 0.82
    eval_path.write_text(json.dumps(eval_data), encoding="utf-8")
    result = evaluate_model_gate(str(eval_path), str(baseline_path), str(config_path))
    assert result["passed"] is False
    assert any("Macro-F1" in r for r in result["reasons"])

    # 3. Reject on latency SLA violation
    eval_data["macro_f1"] = 0.91
    eval_data["mean_latency_ms"] = 18.2
    eval_path.write_text(json.dumps(eval_data), encoding="utf-8")
    result = evaluate_model_gate(str(eval_path), str(baseline_path), str(config_path))
    assert result["passed"] is False
    assert any("latency SLA" in r for r in result["reasons"])

    # 4. Reject on per-class recall floor violation
    eval_data["mean_latency_ms"] = 10.5
    eval_data["per_class"]["0"]["recall"] = 0.75  # Floor is 0.80
    eval_path.write_text(json.dumps(eval_data), encoding="utf-8")
    result = evaluate_model_gate(str(eval_path), str(baseline_path), str(config_path))
    assert result["passed"] is False
    assert any("Class 0 recall" in r for r in result["reasons"])


def test_champion_model_config_schema():
    config_file = Path("configs/champion_model.yaml")
    assert config_file.is_file(), "configs/champion_model.yaml must exist after promotion"
    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["model_name"] == "VehicleClassifier"
    assert data["alias"] == "champion"
    assert "version" in data
    assert "promoted_at" in data
    assert "tracking_uri" in data
    assert data["quality_gate"]["passed"] is True
    assert data["metrics"]["accuracy"] is not None
    assert data["metrics"]["macro_f1"] is not None
