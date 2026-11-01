"""Dataset construction, split generation, and deterministic loaders."""

import logging
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .model import CLASS_IDX
from .preprocessing import inference_transform, training_transform
from .data_versioning import split_indices

logger = logging.getLogger(__name__)


def validate_training_class_mapping(dataset: ImageFolder) -> None:
    expected = ["0_Bus", "1_Truck", "2_Car", "3_Bike", "4_None"]
    if dataset.classes != expected:
        raise ValueError(
            f"Training folders must be {expected} in canonical order; found {dataset.classes}"
        )
    expected_mapping = {name: index for index, name in enumerate(expected)}
    if dataset.class_to_idx != expected_mapping:
        raise ValueError(f"Unexpected ImageFolder class mapping: {dataset.class_to_idx}")


def make_train_validation_loaders(
    dataset_dir,
    *,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    num_workers: int = 0,
    augmentation_config: dict | None = None,
    preprocessing_config: dict | None = None,
    manifest_path=None,
):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_base = ImageFolder(
        dataset_dir,
        transform=training_transform(augmentation_config, preprocessing_config),
    )
    validation_base = ImageFolder(dataset_dir, transform=inference_transform(preprocessing_config))
    validate_training_class_mapping(train_base)
    if train_base.samples != validation_base.samples:
        raise ValueError("Training and validation datasets resolved different files or labels")

    group_ids = None
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest_hashes = {
            row["path"]: row["sha256"]
            for row in manifest["records"]
            if row["dataset"] == "training"
        }
        group_ids = []
        for image_path, _ in train_base.samples:
            relative_path = Path(image_path).relative_to(Path(dataset_dir)).as_posix()
            try:
                group_ids.append(manifest_hashes[relative_path])
            except KeyError as exc:
                raise ValueError(
                    f"Training image {relative_path} is missing from manifest {manifest_path}"
                ) from exc
    validation_indices, train_indices = split_indices(
        len(train_base), seed, validation_fraction, group_ids=group_ids
    )
    train_subset = Subset(train_base, train_indices)
    validation_subset = Subset(validation_base, validation_indices)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator
    )
    validation_loader = DataLoader(
        validation_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    logger.info(
        "Prepared deterministic dataset split",
        extra={"samples": len(train_base), "train": len(train_subset), "validation": len(validation_subset), "seed": seed},
    )
    return train_loader, validation_loader


def class_name_to_index():
    return {name: index for index, name in CLASS_IDX.items()}
