#!/usr/bin/env python3
"""Assign the production alias only after explicit human approval."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="VehicleClassifier")
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--approve", action="store_true", help="Explicitly approve this candidate for the alias")
    args = parser.parse_args()
    if not args.approve:
        parser.error("promotion requires --approve after reviewing the candidate and its gate report")

    import mlflow
    from mlflow import MlflowClient

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version(args.model_name, args.version)
    tags = version.tags or {}
    if tags.get("quality_gate") != "passed" or tags.get("approval_status") != "candidate":
        raise SystemExit("Refusing promotion: this model version is not a gate-passing candidate.")
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    client.set_model_version_tag(args.model_name, args.version, "approval_status", "approved")
    print(f"Promoted {args.model_name} version {args.version} to alias '{args.alias}'.")


if __name__ == "__main__":
    main()
