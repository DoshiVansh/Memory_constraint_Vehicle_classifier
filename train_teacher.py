"""Configurable EfficientNet-B2 teacher training entry point."""

import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from vehicle_classifier_core.config import load_config
from vehicle_classifier_core.data import make_train_validation_loaders
from vehicle_classifier_core.reproducibility import configure_logging, resolve_device, seed_everything, write_run_metadata
from vehicle_classifier_core.tracking import log_epoch_metrics, tracked_run

logger = logging.getLogger(__name__)


@tracked_run("teacher_training")
def train_teacher(config_path="configs/teacher.yaml"):
    config = load_config(config_path)
    configure_logging()
    seed_everything(config["seed"], config.get("deterministic", True))
    device = resolve_device(config.get("device", "auto"))
    logger.info("Using training device", extra={"device": str(device)})
    train_cfg = config["training"]
    data_cfg = config["data"]
    train_loader, validation_loader = make_train_validation_loaders(
        data_cfg["train_dir"],
        batch_size=train_cfg["batch_size"],
        validation_fraction=data_cfg["validation_fraction"],
        seed=config["seed"],
        num_workers=data_cfg.get("num_workers", 0),
        augmentation_config=config.get("augmentation"),
        preprocessing_config=config.get("preprocessing"),
        manifest_path=data_cfg.get("manifest"),
    )
    weights = EfficientNet_B2_Weights.DEFAULT if train_cfg.get("use_imagenet_pretrained") else None
    model = efficientnet_b2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    accumulation = train_cfg.get("gradient_accumulation_steps", 1)
    best_accuracy = -1.0
    checkpoint = train_cfg["checkpoint"]

    for epoch in range(train_cfg["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_batches = 0
        for index, (images, labels) in enumerate(train_loader):
            if train_cfg.get("max_train_batches") and index >= train_cfg["max_train_batches"]:
                break
            logits = model(images.to(device))
            labels = labels.to(device)
            loss = criterion(logits, labels)
            (loss / accumulation).backward()
            if (
                (index + 1) % accumulation == 0
                or index + 1 == len(train_loader)
                or (train_cfg.get("max_train_batches") and index + 1 == train_cfg["max_train_batches"])
            ):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.detach())
            train_batches += 1

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for index, (images, labels) in enumerate(validation_loader):
                if train_cfg.get("max_validation_batches") and index >= train_cfg["max_validation_batches"]:
                    break
                logits = model(images.to(device))
                labels = labels.to(device)
                correct += int((logits.argmax(1) == labels).sum())
                total += len(labels)
        accuracy = correct / total if total else 0.0
        logger.info(
            "Teacher epoch complete",
            extra={"epoch": epoch + 1, "mean_train_loss": train_loss / max(train_batches, 1), "validation_accuracy": accuracy},
        )
        log_epoch_metrics(
            {"train_loss": train_loss / max(train_batches, 1), "validation_accuracy": accuracy},
            epoch + 1,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint)
            metadata_path = config.get("output", {}).get("run_metadata")
            if metadata_path:
                write_run_metadata(metadata_path, config, checkpoint, selected_device=device)
    return {"best_validation_accuracy": best_accuracy, "checkpoint": checkpoint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/teacher.yaml")
    args = parser.parse_args()
    configure_logging()
    try:
        train_teacher(args.config)
    except Exception:
        logger.exception("Teacher training failed", extra={"config": args.config})
        raise


if __name__ == "__main__":
    main()
