"""Local-first MLflow tracking helpers for training and evaluation runs."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from .config import load_config
from .model import CLASS_IDX
from .reproducibility import file_sha256, git_code_working_tree_dirty, git_revision

logger = logging.getLogger(__name__)


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for experiment tracking. Install it with "
            "`python -m pip install -r requirements-mlflow.txt`."
        ) from exc
    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    return mlflow


def _lineage(config: dict) -> dict:
    manifest = config.get("data_manifest") or config.get("data", {}).get("manifest")
    manifest_path = Path(manifest) if manifest else None
    lock_path = Path("dvc.lock")
    lock_sha = file_sha256(lock_path) if lock_path.is_file() else None
    data_version = os.environ.get("DVC_DATA_VERSION")
    manifest_sha = None
    if manifest_path and manifest_path.is_file():
        manifest_sha = file_sha256(manifest_path)
        data_version = data_version or f"manifest-sha256:{manifest_sha}"
    elif lock_sha:
        data_version = data_version or f"dvc-lock-sha256:{lock_sha}"
    data_version = data_version or "unversioned-local-data"

    train_cfg = config.get("training", {})
    teacher_path = train_cfg.get("teacher_checkpoint")
    teacher_sha = file_sha256(teacher_path) if teacher_path and Path(teacher_path).is_file() else None
    return {
        "git_sha": git_revision() or "unknown",
        "git_dirty": git_code_working_tree_dirty(),
        "data_version": data_version,
        "data_manifest_sha256": manifest_sha or "unavailable",
        "dvc_lock_sha256": lock_sha or "unavailable",
        "teacher_identifier": "efficientnet_b2" if teacher_path else "not_applicable",
        "teacher_checkpoint_sha256": teacher_sha or "unavailable",
    }


def _safe_param(value, maximum: int = 450) -> str:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, default=str)
    elif value is None:
        value = "null"
    else:
        value = str(value)
    return value if len(value) <= maximum else value[: maximum - 3] + "..."


def _source_snapshot(output_path: Path) -> dict:
    """Package versioned source/config files so dirty worktrees remain identifiable."""
    raw_paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        stderr=subprocess.DEVNULL,
    ).decode("utf-8", errors="surrogateescape").split("\0")
    exact_names = {".gitignore", ".dvcignore", "dvc.yaml", "dvc.lock", "LICENSE"}
    source_suffixes = {".py", ".yaml", ".yml", ".toml", ".sh", ".md", ".txt", ".dvc"}
    files = []
    for name in sorted(set(filter(None, raw_paths))):
        path = Path(name)
        if name in exact_names or path.name in exact_names or path.suffix in source_suffixes:
            if path.is_file():
                files.append((name, path.read_bytes()))
    if not files:
        raise RuntimeError("Cannot create a source snapshot because no source files were found")
    entries = [{"path": name, "sha256": hashlib.sha256(content).hexdigest()} for name, content in files]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot_sha = hashlib.sha256(canonical).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in files:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return {"sha256": snapshot_sha, "file_count": len(files), "archive": output_path}


def log_epoch_metrics(metrics: dict, step: int) -> None:
    """Log scalar epoch values when a tracked training run is active."""
    mlflow = _mlflow()
    if mlflow.active_run() is None:
        return
    values = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    if values:
        mlflow.log_metrics(values, step=step)


def tracked_run(kind: str):
    """Decorate a CLI-callable workflow so each invocation creates an MLflow run."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(config_path="configs/distillation.yaml", *args, **kwargs):
            config = load_config(config_path)
            mlflow = _mlflow()
            experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "VehicleClassifier")
            mlflow.set_experiment(experiment_name)
            lineage = _lineage(config)
            snapshot_dir = tempfile.TemporaryDirectory(prefix="vehicle-classifier-source-")
            snapshot = _source_snapshot(Path(snapshot_dir.name) / "source_snapshot.zip")
            run_name = f"{kind}-{Path(config_path).stem}"
            tags = {
                "workflow": kind,
                "git_sha": lineage["git_sha"],
                "git_dirty": str(lineage["git_dirty"]),
                "source_snapshot_sha256": snapshot["sha256"],
                "data_version": lineage["data_version"],
                "data_manifest_sha256": lineage["data_manifest_sha256"],
                "dvc_lock_sha256": lineage["dvc_lock_sha256"],
                "teacher_identifier": "efficientnet_b2" if kind == "teacher_training" or lineage["teacher_identifier"] != "not_applicable" else "not_applicable",
                "teacher_checkpoint_sha256": lineage["teacher_checkpoint_sha256"],
                "student_identifier": "shufflenet_v2_x1_0" if kind != "teacher_training" else "not_applicable",
                "teacher_initialization": (
                    "imagenet_pretrained" if config.get("training", {}).get("use_imagenet_pretrained") else "configured_checkpoint"
                ) if kind == "teacher_training" else "checkpoint",
                "class_mapping": json.dumps({str(k): v for k, v in CLASS_IDX.items()}, sort_keys=True),
            }
            with mlflow.start_run(run_name=run_name, tags=tags) as run:
                flat_params = {"seed": config.get("seed"), "device": config.get("device", "auto")}
                for section, values in config.items():
                    if isinstance(values, dict):
                        flat_params.update({f"{section}.{key}": value for key, value in values.items()})
                mlflow.log_params({key: _safe_param(value) for key, value in flat_params.items() if value is not None})
                mlflow.log_dict(config, "configuration.yaml.json")
                mlflow.log_artifact(str(snapshot["archive"]), artifact_path="lineage")
                mlflow.set_tag("source_snapshot_file_count", str(snapshot["file_count"]))
                try:
                    result = function(config_path, *args, **kwargs)
                    if isinstance(result, dict):
                        result["mlflow_run_id"] = run.info.run_id
                        result["source_snapshot_sha256"] = snapshot["sha256"]
                        mlflow.log_metrics({
                            key: float(value)
                            for key, value in result.items()
                            if isinstance(value, (int, float)) and value is not None
                        })
                        checkpoint = result.get("checkpoint")
                        if checkpoint and Path(checkpoint).is_file():
                            mlflow.log_artifact(str(checkpoint), artifact_path="checkpoint")
                            metadata_path = Path(str(checkpoint) + ".json")
                            if metadata_path.is_file():
                                mlflow.log_artifact(str(metadata_path), artifact_path="checkpoint")
                        report_path = config.get("output", {}).get("run_metadata")
                        if report_path and Path(report_path).is_file():
                            metadata = json.loads(Path(report_path).read_text(encoding="utf-8"))
                            metadata["mlflow_run_id"] = run.info.run_id
                            Path(report_path).write_text(
                                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                            )
                            mlflow.log_artifact(report_path, artifact_path="run_metadata")
                    return result
                except Exception as exc:
                    mlflow.set_tag("run_status", "failed")
                    mlflow.set_tag("failure", _safe_param(f"{type(exc).__name__}: {exc}"))
                    raise

        return wrapped

    return decorate


def tracked_evaluation(function):
    """Track evaluation output, metrics, confusion data, and exact checkpoint bytes."""

    @functools.wraps(function)
    def wrapped(config_path="configs/evaluation.yaml", *args, **kwargs):
        config = load_config(config_path)
        mlflow = _mlflow()
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "VehicleClassifier"))
        lineage = _lineage(config)
        snapshot_dir = tempfile.TemporaryDirectory(prefix="vehicle-classifier-source-")
        snapshot = _source_snapshot(Path(snapshot_dir.name) / "source_snapshot.zip")
        checkpoint = Path(config["checkpoint"])
        tags = {
            "workflow": "evaluation",
            "git_sha": lineage["git_sha"],
            "git_dirty": str(lineage["git_dirty"]),
            "source_snapshot_sha256": snapshot["sha256"],
            "data_version": lineage["data_version"],
            "data_manifest_sha256": lineage["data_manifest_sha256"],
            "dvc_lock_sha256": lineage["dvc_lock_sha256"],
            "student_identifier": "shufflenet_v2_x1_0",
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.is_file() else "missing",
        }
        with mlflow.start_run(run_name=f"evaluation-{Path(config_path).stem}", tags=tags) as run:
            mlflow.log_params({"seed": config.get("seed", "unknown"), "dataset_dir": config["dataset_dir"]})
            mlflow.log_dict(config, "configuration.yaml.json")
            mlflow.log_artifact(str(snapshot["archive"]), artifact_path="lineage")
            mlflow.set_tag("source_snapshot_file_count", str(snapshot["file_count"]))
            report = function(config_path, *args, **kwargs)
            report["run"]["mlflow_run_id"] = run.info.run_id
            report["run"]["data_version"] = lineage["data_version"]
            report["run"]["dvc_lock_sha256"] = lineage["dvc_lock_sha256"]
            report["run"]["git_working_tree_dirty"] = lineage["git_dirty"]
            report["run"]["source_snapshot_sha256"] = snapshot["sha256"]
            output = Path(config["output"])
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            metric_values = {}
            for key in ("accuracy", "elapsed_seconds", "mean_latency_ms"):
                value = report.get(key)
                if isinstance(value, (int, float)):
                    metric_values[key] = float(value)
            for subset in ("clean", "augmented"):
                value = report.get(subset, {}).get("accuracy")
                if isinstance(value, (int, float)):
                    metric_values[f"{subset}_accuracy"] = float(value)
            for index, item in report.get("per_class", {}).items():
                if item.get("accuracy") is not None:
                    metric_values[f"class_{index}_accuracy"] = float(item["accuracy"])
            mlflow.log_metrics(metric_values)
            mlflow.log_dict(report, "evaluation.json")
            if checkpoint.is_file():
                from .mlflow_model import VehicleClassifierPyfunc

                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=VehicleClassifierPyfunc(),
                    artifacts={"checkpoint": str(checkpoint.resolve())},
                    code_paths=[str(Path(__file__).resolve().parent)],
                    pip_requirements="requirements-mlflow-model.txt",
                )
                sidecar = Path(str(checkpoint) + ".json")
                if sidecar.is_file():
                    mlflow.log_artifact(str(sidecar), artifact_path="model_metadata")
            mlflow.log_dict(report.get("confusion_matrix", []), "confusion_matrix.json")
            mlflow.log_artifact(str(output), artifact_path="reports")
            return report

    return wrapped
