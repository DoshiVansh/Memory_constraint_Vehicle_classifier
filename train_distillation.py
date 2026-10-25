"""Configurable ShuffleNet V2 knowledge-distillation training entry point."""

import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import efficientnet_b2

from vehicle_classifier_core.config import load_config
from vehicle_classifier_core.data import make_train_validation_loaders
from vehicle_classifier_core.model import StudentShuffleNet
from vehicle_classifier_core.packaging import save_checkpoint
from vehicle_classifier_core.reproducibility import configure_logging, resolve_device, seed_everything, write_run_metadata
from vehicle_classifier_core.tracking import log_epoch_metrics, tracked_run

logger = logging.getLogger(__name__)


@tracked_run("student_distillation")
def train_distillation(config_path="configs/distillation.yaml"):
    config = load_config(config_path)
    configure_logging()
    seed_everything(config["seed"], config.get("deterministic", True))
    device = resolve_device(config.get("device", "auto"))
    logger.info("Using training device", extra={"device": str(device)})

    data_cfg = config["data"]
    train_loader, validation_loader = make_train_validation_loaders(
        data_cfg["train_dir"],
        batch_size=config["training"]["batch_size"],
        validation_fraction=data_cfg["validation_fraction"],
        seed=config["seed"],
        num_workers=data_cfg.get("num_workers", 0),
        augmentation_config=config.get("augmentation"),
        preprocessing_config=config.get("preprocessing"),
    )

    teacher = efficientnet_b2(weights=None)
    teacher.classifier[1] = nn.Linear(teacher.classifier[1].in_features, 5)
    teacher_checkpoint = config["training"]["teacher_checkpoint"]
    teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device, weights_only=True))
    teacher.to(device).eval()

    student = StudentShuffleNet(num_classes=5).to(device)
    optimizer = optim.Adam(student.parameters(), lr=config["training"]["learning_rate"])
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction="batchmean")
    alpha = config["training"]["alpha"]
    temperature = config["training"]["temperature"]
    best_accuracy = -1.0
    checkpoint_path = config["training"]["checkpoint"]
    output_cfg = config.get("output", {})

    for epoch in range(config["training"]["epochs"]):
        epoch_start = time.perf_counter()
        student.train()
        loss_sum = 0.0
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = alpha * ce_loss(student_logits, labels) + (1 - alpha) * kd_loss(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
            ) * temperature**2
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += int((student_logits.argmax(1) == labels).sum())
            total += len(labels)

        student.eval()
        val_correct = val_total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                logits = student(images.to(device))
                labels = labels.to(device)
                val_correct += int((logits.argmax(1) == labels).sum())
                val_total += len(labels)
        accuracy = val_correct / val_total if val_total else 0.0
        logger.info(
            "Epoch complete",
            extra={
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(len(train_loader), 1),
                "train_accuracy": correct / max(total, 1),
                "validation_accuracy": accuracy,
                "elapsed_seconds": round(time.perf_counter() - epoch_start, 3),
            },
        )
        log_epoch_metrics(
            {
                "train_loss": loss_sum / max(len(train_loader), 1),
                "train_accuracy": correct / max(total, 1),
                "validation_accuracy": accuracy,
            },
            epoch + 1,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            student.half()
            metadata = save_checkpoint(
                student,
                checkpoint_path,
                config=config,
                metrics={"validation_accuracy": accuracy, "epoch": epoch + 1},
            )
            student.float()
            if output_cfg.get("run_metadata"):
                write_run_metadata(output_cfg["run_metadata"], config, checkpoint_path, selected_device=device)
            if metadata["size_bytes"] >= config["artifact"]["max_size_bytes"]:
                raise RuntimeError(
                    f"Checkpoint is {metadata['size_bytes']} bytes; limit is "
                    f"{config['artifact']['max_size_bytes']} bytes"
                )
    return {"best_validation_accuracy": best_accuracy, "checkpoint": checkpoint_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distillation.yaml")
    args = parser.parse_args()
    try:
        train_distillation(args.config)
    except Exception:
        configure_logging(logging.INFO)
        logger.exception("Distillation training failed", extra={"config": args.config})
        raise


if __name__ == "__main__":
    main()
