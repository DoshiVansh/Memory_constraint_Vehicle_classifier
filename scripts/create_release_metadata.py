#!/usr/bin/env python3
"""Validate the approved model package and write image release lineage metadata."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

MAX_MODEL_BYTES = 5_000_000
IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_VERSION_PATTERN = re.compile(r"^[1-9][0-9]*$")
EXPECTED_CLASS_MAPPING = {"0": "Bus", "1": "Truck", "2": "Car", "3": "Bike", "4": "None"}
EXPECTED_PREPROCESSING = {
    "color": "RGB",
    "size": [224, 224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def validate_model_package(checkpoint_path, metadata_path):
    checkpoint = Path(checkpoint_path)
    metadata_file = Path(metadata_path)
    if not checkpoint.is_file() or not metadata_file.is_file():
        raise ValueError("The checkpoint and its JSON sidecar must both exist")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint metadata: {exc}") from exc

    size_bytes = checkpoint.stat().st_size
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if size_bytes <= 0 or size_bytes >= MAX_MODEL_BYTES:
        raise ValueError(f"Checkpoint must be smaller than {MAX_MODEL_BYTES} bytes")
    if metadata.get("size_bytes") != size_bytes or metadata.get("sha256") != digest:
        raise ValueError("Checkpoint size or SHA-256 differs from the sidecar metadata")
    if metadata.get("architecture") != "shufflenet_v2_x1_0":
        raise ValueError("Release checkpoint architecture is not the approved ShuffleNet student")
    if metadata.get("class_mapping") != EXPECTED_CLASS_MAPPING:
        raise ValueError("Release checkpoint class mapping differs from the inference API contract")
    if metadata.get("preprocessing") != EXPECTED_PREPROCESSING:
        raise ValueError("Release checkpoint preprocessing differs from the inference API contract")
    if metadata.get("model_alias") != "champion":
        raise ValueError("Release metadata must identify the approved champion model alias")
    if not metadata.get("model_name") or not metadata.get("model_version"):
        raise ValueError("Release metadata must include the registered model name and version")
    if not MODEL_NAME_PATTERN.fullmatch(str(metadata["model_name"])):
        raise ValueError("Registered model name contains unsupported characters")
    if not MODEL_VERSION_PATTERN.fullmatch(str(metadata["model_version"])):
        raise ValueError("Registered model version must be a positive integer")
    return {
        "name": str(metadata["model_name"]),
        "version": str(metadata["model_version"]),
        "alias": str(metadata["model_alias"]),
        "artifact_sha256": digest,
        "artifact_size_bytes": size_bytes,
        "architecture": metadata["architecture"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("student_model.pth"))
    parser.add_argument("--model-metadata", type=Path, default=Path("student_model.pth.json"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--git-sha")
    parser.add_argument("--release-tag")
    parser.add_argument("--image-name")
    parser.add_argument("--image-digest")
    parser.add_argument("--output", type=Path, default=Path("reports/release_metadata.json"))
    args = parser.parse_args()

    try:
        model = validate_model_package(args.checkpoint, args.model_metadata)
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as output:
                for key, value in {
                    "model_name": model["name"],
                    "model_version": model["version"],
                    "model_alias": model["alias"],
                    "artifact_sha256": model["artifact_sha256"],
                    "artifact_size_bytes": model["artifact_size_bytes"],
                }.items():
                    output.write(f"{key}={value}\n")
        if args.validate_only:
            print(json.dumps(model, sort_keys=True))
            return

        missing = [name for name in ("git_sha", "release_tag", "image_name", "image_digest") if not getattr(args, name)]
        if missing:
            parser.error("release metadata requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
        if not IMAGE_DIGEST_PATTERN.fullmatch(args.image_digest):
            parser.error("--image-digest must have the form sha256:<64 lowercase hex characters>")

        image_reference = f"{args.image_name}@{args.image_digest}"
        release_metadata = {
            "schema_version": 1,
            "release_tag": args.release_tag,
            "git_sha": args.git_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "image": {
                "name": args.image_name,
                "digest": args.image_digest,
                "immutable_reference": image_reference,
                "release_tag": args.release_tag,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(release_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote release lineage metadata to {args.output}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
