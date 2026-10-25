#!/usr/bin/env python3
"""Register a gate-passing evaluation artifact as an MLflow candidate version."""

import argparse
import json
import os
from pathlib import Path

from vehicle_classifier_core.model_gate import evaluate_model_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="reports/evaluation.json")
    parser.add_argument("--baseline", default="reports/baseline.json")
    parser.add_argument("--gate-config", default="configs/quality_gates.yaml")
    parser.add_argument("--model-name", default="VehicleClassifier")
    args = parser.parse_args()
    gate = evaluate_model_gate(args.evaluation, args.baseline, args.gate_config)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    if not gate["passed"]:
        print("Candidate not registered: quality gate failed.")
        print("; ".join(gate["reasons"]))
        raise SystemExit(1)

    import mlflow
    from mlflow import MlflowClient

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    run_id = gate["evaluation"]["mlflow_run_id"]
    version = mlflow.register_model(f"runs:/{run_id}/model", args.model_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_model_version_tag(args.model_name, version.version, "approval_status", "candidate")
    client.set_model_version_tag(args.model_name, version.version, "quality_gate", "passed")
    client.set_model_version_tag(args.model_name, version.version, "git_sha", gate["evaluation"]["git_revision"])
    client.set_model_version_tag(args.model_name, version.version, "data_version", gate["evaluation"]["data_version"] or "unknown")
    client.set_model_version_tag(args.model_name, version.version, "artifact_sha256", gate["artifact"]["sha256"])
    client.log_artifact(run_id, "reports/model_gate.json", artifact_path="quality_gate")
    print(f"Registered {args.model_name} version {version.version} as a candidate.")
    print("Promotion requires the separate scripts/promote_model.py --approve command.")


if __name__ == "__main__":
    main()
