"""Validate an inference checkpoint and its release metadata before serving."""

import json
from pathlib import Path

from .model import CLASS_IDX
from .preprocessing import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from .reproducibility import file_sha256

MAX_CHECKPOINT_BYTES = 5_000_000
ARCHITECTURE = "shufflenet_v2_x1_0"


def validate_inference_package(checkpoint_path, metadata_path=None):
    checkpoint = Path(checkpoint_path)
    metadata_file = Path(metadata_path) if metadata_path else Path(f"{checkpoint}.json")
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    if not metadata_file.is_file():
        raise ValueError(f"Checkpoint metadata does not exist: {metadata_file}")

    size_bytes = checkpoint.stat().st_size
    if size_bytes <= 0 or size_bytes >= MAX_CHECKPOINT_BYTES:
        raise ValueError(f"Checkpoint size must be between 1 and {MAX_CHECKPOINT_BYTES - 1} bytes")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint metadata: {exc}") from exc

    expected_classes = {str(index): label for index, label in CLASS_IDX.items()}
    if metadata.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Unsupported checkpoint architecture: {metadata.get('architecture')!r}")
    if metadata.get("class_mapping") != expected_classes:
        raise ValueError("Checkpoint class mapping does not match this service")
    expected_preprocessing = {
        "color": "RGB",
        "size": list(IMAGE_SIZE),
        "mean": list(NORMALIZE_MEAN),
        "std": list(NORMALIZE_STD),
    }
    if metadata.get("preprocessing") != expected_preprocessing:
        raise ValueError("Checkpoint preprocessing metadata does not match this service")
    if metadata.get("size_bytes") != size_bytes:
        raise ValueError("Checkpoint size does not match its metadata")
    actual_sha256 = file_sha256(checkpoint)
    if metadata.get("sha256") != actual_sha256:
        raise ValueError("Checkpoint SHA-256 does not match its metadata")
    return {
        "path": str(checkpoint),
        "sha256": actual_sha256,
        "size_bytes": size_bytes,
        "model_version": str(metadata.get("model_version", "unknown")),
        "architecture": metadata["architecture"],
    }
