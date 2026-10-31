#!/usr/bin/env python3
"""Assign the production alias and record declarative champion config after explicit human approval."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import yaml

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="VehicleClassifier")
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument(
        "--gate-report",
        default="reports/model_gate.json",
        help="Path to model quality gate verification report",
    )
    parser.add_argument(
        "--champion-config",
        default="configs/champion_model.yaml",
        help="Path to write declarative champion model tracking record",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Explicitly approve this candidate for the alias",
    )
    args = parser.parse_args()
    if not args.approve:
        parser.error("promotion requires --approve after reviewing the candidate and its gate report")

    # Verify model quality gate report
    gate_report_path = Path(args.gate_report)
    if not gate_report_path.is_file():
        raise SystemExit(f"Refusing promotion: quality gate report not found at {args.gate_report}")
    gate_data = json.loads(gate_report_path.read_text(encoding="utf-8"))
    if not gate_data.get("passed"):
        reasons = "; ".join(gate_data.get("reasons", []))
        raise SystemExit(f"Refusing promotion: quality gate did not pass ({reasons})")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if "MLFLOW_TRACKING_USERNAME" not in os.environ and "DAGSHUB_USERNAME" in os.environ:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    if "MLFLOW_TRACKING_PASSWORD" not in os.environ and "DAGSHUB_TOKEN" in os.environ:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

    import mlflow
    from mlflow import MlflowClient

    default_uri = "sqlite:///mlflow.db"
    if "DAGSHUB_USERNAME" in os.environ and "DAGSHUB_REPO_NAME" in os.environ:
        default_uri = f"https://dagshub.com/{os.environ['DAGSHUB_USERNAME']}/{os.environ['DAGSHUB_REPO_NAME']}.mlflow"

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", default_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    version_obj = client.get_model_version(args.model_name, args.version)
    tags = version_obj.tags or {}
    if tags.get("quality_gate") == "failed":
        raise SystemExit("Refusing promotion: this model version is marked with quality_gate=failed in MLflow.")

    promoted_at = datetime.now(timezone.utc).isoformat()
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    client.set_model_version_tag(args.model_name, args.version, "approval_status", "approved")
    client.set_model_version_tag(args.model_name, args.version, "promoted_at", promoted_at)
    print(f"Promoted {args.model_name} version {args.version} to alias '{args.alias}' on {tracking_uri}.")

    # Write declarative GitOps champion record
    eval_info = gate_data.get("evaluation", {})
    artifact_info = gate_data.get("artifact", {})
    champion_data = {
        "model_name": args.model_name,
        "version": str(args.version),
        "alias": args.alias,
        "promoted_at": promoted_at,
        "tracking_uri": tracking_uri,
        "run_id": eval_info.get("mlflow_run_id"),
        "git_revision": eval_info.get("git_revision"),
        "source_snapshot_sha256": eval_info.get("source_snapshot_sha256"),
        "data_version": eval_info.get("data_version"),
        "artifact": {
            "path": artifact_info.get("path"),
            "sha256": artifact_info.get("sha256"),
            "size_bytes": artifact_info.get("size_bytes"),
        },
        "metrics": {
            "accuracy": eval_info.get("accuracy"),
            "macro_f1": eval_info.get("macro_f1"),
            "mean_latency_ms": eval_info.get("mean_latency_ms"),
        },
        "quality_gate": {
            "passed": True,
            "gate_report": str(args.gate_report),
            "thresholds": gate_data.get("thresholds", {}),
        },
    }

    champion_path = Path(args.champion_config)
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    with champion_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(champion_data, f, sort_keys=False)
    print(f"Recorded declarative champion model tracking to {champion_path}.")

    # Keep sidecar JSON consistent if it exists
    sidecar_path = Path("student_model.pth.json")
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["model_version"] = str(args.version)
            sidecar["model_alias"] = args.alias
            sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    main()
