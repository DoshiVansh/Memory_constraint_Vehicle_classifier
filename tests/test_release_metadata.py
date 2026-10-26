import hashlib
import json
import sys

import pytest

from scripts.create_release_metadata import main, validate_model_package

APPROVED_SIDECAR_FIELDS = {
    "architecture": "shufflenet_v2_x1_0",
    "model_name": "VehicleClassifier",
    "model_version": "2",
    "model_alias": "champion",
    "class_mapping": {"0": "Bus", "1": "Truck", "2": "Car", "3": "Bike", "4": "None"},
    "preprocessing": {
        "color": "RGB",
        "size": [224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def write_model_package(directory, *, alias="champion"):
    checkpoint = directory / "model.pth"
    checkpoint.write_bytes(b"approved-checkpoint-fixture")
    sidecar = directory / "model.pth.json"
    metadata = {
        **APPROVED_SIDECAR_FIELDS,
        "model_alias": alias,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "size_bytes": checkpoint.stat().st_size,
    }
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return checkpoint, sidecar


def test_release_metadata_requires_approved_exact_artifact(tmp_path):
    checkpoint, sidecar = write_model_package(tmp_path)
    model = validate_model_package(checkpoint, sidecar)
    assert model["version"] == "2"
    assert model["alias"] == "champion"
    assert model["artifact_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_release_metadata_rejects_changed_artifact_and_candidate_alias(tmp_path):
    checkpoint, sidecar = write_model_package(tmp_path)
    checkpoint.write_bytes(b"changed-checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_model_package(checkpoint, sidecar)

    checkpoint, sidecar = write_model_package(tmp_path, alias="candidate")
    with pytest.raises(ValueError, match="champion"):
        validate_model_package(checkpoint, sidecar)


def test_release_metadata_binds_commit_model_and_image_digest(tmp_path, monkeypatch):
    checkpoint, sidecar = write_model_package(tmp_path)
    output = tmp_path / "release_metadata.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_release_metadata.py",
            "--checkpoint", str(checkpoint),
            "--model-metadata", str(sidecar),
            "--git-sha", "abc123",
            "--release-tag", "v1.2.3",
            "--image-name", "ghcr.io/example/vehicle-classifier",
            "--image-digest", "sha256:" + "a" * 64,
            "--output", str(output),
        ],
    )
    main()
    metadata = json.loads(output.read_text(encoding="utf-8"))
    assert metadata["git_sha"] == "abc123"
    assert metadata["model"]["version"] == "2"
    assert metadata["model"]["artifact_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["image"]["immutable_reference"].endswith("sha256:" + "a" * 64)
