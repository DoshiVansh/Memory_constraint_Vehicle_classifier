"""Evaluate a checkpoint against a labeled folder dataset and write JSON metrics."""

import argparse
import json
import logging
import time
from pathlib import Path

from vehicle_classifier_core.config import load_config
from vehicle_classifier_core.evaluation import classification_report
from vehicle_classifier_core.predictor import VehicleClassifier
from vehicle_classifier_core.reproducibility import configure_logging, file_sha256, git_revision, run_metadata
from vehicle_classifier_core.tracking import tracked_evaluation

logger = logging.getLogger(__name__)


@tracked_evaluation
def evaluate(config_path="configs/evaluation.yaml"):
    config = load_config(config_path)
    classifier = VehicleClassifier(config["checkpoint"], config.get("preprocessing"))
    dataset_dir = Path(config["dataset_dir"])
    class_mapping = {int(index): name for name, index in config["classes"].items()}
    targets, predictions = [], []
    clean_targets, clean_predictions = [], []
    augmented_targets, augmented_predictions = [], []
    start = time.perf_counter()
    for true_index, class_name in class_mapping.items():
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            predicted = classifier.predict(str(image_path))
            targets.append(true_index)
            predictions.append(predicted)
            if image_path.name.startswith("clean"):
                clean_targets.append(true_index)
                clean_predictions.append(predicted)
            elif image_path.name.startswith("aug"):
                augmented_targets.append(true_index)
                augmented_predictions.append(predicted)
    elapsed = time.perf_counter() - start
    report = classification_report(targets, predictions, num_classes=len(class_mapping))
    report["classes"] = [class_mapping[index] for index in range(len(class_mapping))]
    report["clean"] = classification_report(clean_targets, clean_predictions, len(class_mapping))
    report["augmented"] = classification_report(augmented_targets, augmented_predictions, len(class_mapping))
    report["elapsed_seconds"] = elapsed
    report["mean_latency_ms"] = (1000 * elapsed / len(targets)) if targets else None
    report["run"] = {
        "git_revision": git_revision(),
        "config": config,
        "checkpoint_sha256": file_sha256(config["checkpoint"]),
        "checkpoint_size_bytes": Path(config["checkpoint"]).stat().st_size,
        "environment": run_metadata(config, config["checkpoint"]),
    }
    manifest_path = Path(config["data_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}; run `dvc repro` to materialize it first"
        )
    report["run"]["dataset_manifest"] = {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Evaluation complete", extra={"samples": len(targets), "accuracy": report["accuracy"], "output": str(output)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    configure_logging()
    try:
        evaluate(args.config)
    except Exception:
        logger.exception("Evaluation failed", extra={"config": args.config})
        raise


if __name__ == "__main__":
    main()
