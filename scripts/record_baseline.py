#!/usr/bin/env python3
"""Record a fresh evaluation only when it matches the audited canonical checkpoint."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="reports/evaluation.json")
    parser.add_argument("--baseline", default="reports/baseline.json")
    parser.add_argument("--confirm-canonical", action="store_true", help="Confirm this is the agreed canonical baseline run")
    args = parser.parse_args()
    if not args.confirm_canonical:
        parser.error("refreshing the canonical baseline requires --confirm-canonical")

    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    run = evaluation.get("run", {})
    expected_hash = baseline.get("model", {}).get("sha256")
    if run.get("checkpoint_sha256") != expected_hash:
        raise SystemExit("Evaluation checkpoint hash does not match the audited canonical model")
    if not run.get("mlflow_run_id") or not run.get("dataset_manifest", {}).get("sha256"):
        raise SystemExit("Evaluation must have an MLflow run ID and dataset-manifest checksum")
    if not run.get("git_revision") or not run.get("source_snapshot_sha256"):
        raise SystemExit("Baseline evaluation must include a Git revision and exact source snapshot")

    baseline["fresh_run"] = {
        "completed": True,
        "mlflow_run_id": run["mlflow_run_id"],
        "git_revision": run["git_revision"],
        "source_snapshot_sha256": run["source_snapshot_sha256"],
        "data_version": run.get("data_version"),
        "dataset_manifest_sha256": run["dataset_manifest"]["sha256"],
        "dvc_lock_sha256": run.get("dvc_lock_sha256"),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "overall_accuracy": evaluation.get("accuracy"),
        "per_class": evaluation.get("per_class"),
        "confusion_matrix": evaluation.get("confusion_matrix"),
        "sample_count": evaluation.get("sample_count"),
        "mean_latency_ms": evaluation.get("mean_latency_ms"),
    }
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded fresh canonical baseline in {baseline_path}.")


if __name__ == "__main__":
    main()
