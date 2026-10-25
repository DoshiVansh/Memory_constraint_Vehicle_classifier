"""Seed control and run environment metadata."""

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class JsonLogFormatter(logging.Formatter):
    """Emit standard logging fields plus caller-provided context as JSON."""

    _reserved = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in self._reserved and not key.startswith("_")}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("Random generators seeded", extra={"seed": seed, "deterministic": deterministic})


def resolve_device(configured: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda, failing clearly for an unavailable explicit CUDA request."""
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build compatible with your GPU driver, "
            "or set device: cpu."
        )
    return device


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_working_tree_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def git_code_working_tree_dirty() -> bool | None:
    """Report source/config changes while excluding expected generated data/reports."""
    generated = (
        ".dvc/",
        "dvc.lock",
        "data/manifests/",
        "reports/data_validation.json",
        "reports/evaluation.json",
        "reports/ood_evaluation.json",
        "reports/teacher_run.json",
        "reports/distillation_run.json",
        "reports/model_gate.json",
        "student_model.pth",
        "student_model.pth.json",
        "EfficientNet_B2_teacher_model.pth",
    )
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        changed_paths = [line[3:].strip() for line in status.splitlines() if len(line) >= 4]
        return any(not path.startswith(generated) for path in changed_paths)
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(config: dict, checkpoint=None, selected_device=None) -> dict:
    import torchvision
    import yaml
    from PIL import __version__ as pillow_version

    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": pillow_version,
        "pyyaml": yaml.__version__,
        "git_revision": git_revision(),
        "git_working_tree_dirty": git_working_tree_dirty(),
        "configured_device": config.get("device", "auto"),
        "selected_device": str(selected_device) if selected_device is not None else None,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": config,
    }
    if checkpoint is not None:
        metadata["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "size_bytes": Path(checkpoint).stat().st_size,
        }
    return metadata


def write_run_metadata(path, config: dict, checkpoint=None, selected_device=None) -> dict:
    metadata = run_metadata(config, checkpoint=checkpoint, selected_device=selected_device)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
