"""Checkpoint validation and release metadata packaging."""

import json
from pathlib import Path

import torch

from .model import CLASS_IDX
from .reproducibility import file_sha256, git_revision
from .preprocessing import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def save_checkpoint(model, path, *, config: dict, metrics: dict | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    size_bytes = path.stat().st_size
    size_limit = config.get("artifact", {}).get("max_size_bytes")
    if size_limit is not None and size_bytes >= size_limit:
        raise ValueError(f"Serialized checkpoint is {size_bytes} bytes; configured limit is {size_limit}")
    metadata = {
        "architecture": "shufflenet_v2_x1_0",
        "class_mapping": {str(k): v for k, v in CLASS_IDX.items()},
        "preprocessing": config.get("preprocessing", {
            "color": "RGB",
            "size": list(IMAGE_SIZE),
            "mean": list(NORMALIZE_MEAN),
            "std": list(NORMALIZE_STD),
        }),
        "checkpoint": path.name,
        "sha256": file_sha256(path),
        "size_bytes": size_bytes,
        "git_revision": git_revision(),
        "config": config,
        "metrics": metrics or {},
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
