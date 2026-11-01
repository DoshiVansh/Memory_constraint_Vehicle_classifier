"""
scripts/ingest_data.py — Unified DVC Stage 0: Raw ImageNet → Disjoint Train + Test Splits.

This script replaces the two independent standalone scripts:
  - dataset_organizer.py  (→ vehicle_dataset/)
  - build_test_data.py    (→ test_dataset/)

Key improvement: a **shared global exclusion registry** ensures that every raw
ImageNet image assigned to `vehicle_dataset` is permanently excluded from the
pool used by `test_dataset`, guaranteeing zero raw-source overlap between splits.

Outputs:
  - vehicle_dataset/          Training split (max_training_images_per_class per class)
  - test_dataset/             Target evaluation split (images_per_class per class, 50% clean + 50% aug)
  - data/manifests/raw_split_manifest.json   Lineage record linking raw filenames → split + role

Usage (via DVC — do not run directly):
  dvc repro ingest_data

Or manually (for debugging only):
  python scripts/ingest_data.py --config configs/data.yaml
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image
import torchvision.transforms as transforms

from vehicle_classifier_core.config import load_config
from vehicle_classifier_core.reproducibility import seed_everything


# Class → ImageNet synset folder mapping (identical to the original scripts)
CLASS_MAPPING = {
    "Bus":   ["n04487081", "n04146614", "n02924116"],
    "Truck": ["n04467665", "n03417042", "n03345487"],
    "Car":   ["n02958343", "n03100232", "n03594945", "n03769881", "n04037443", "n04285008"],
    "Bike":  ["n03790512", "n02835271"],
    "None":  ["n06874185", "n04328186", "n03881040", "n03710193"],
}


def _collect_synset_images(source_path: Path, synsets: list[str]) -> list[Path]:
    """Return all valid image paths across the given synset folders, sorted for determinism."""
    images: list[Path] = []
    for synset in synsets:
        folder = source_path / synset
        if folder.exists():
            images.extend(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        else:
            print(f"  Warning: synset folder '{synset}' not found in {source_path}")
    return sorted(images)  # sort for deterministic shuffle seeding


def _build_augmentation_pipeline(corruption: dict, image_size: tuple[int, int]) -> transforms.Compose:
    """Build the augmentation transform used for the 'aug_' half of the test split."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=corruption["horizontal_flip_probability"]),
        transforms.RandomAffine(
            degrees=corruption["rotation_degrees"],
            translate=tuple(corruption["translation_fraction"]),
            scale=tuple(corruption["scale_range"]),
        ),
        transforms.ColorJitter(
            brightness=corruption["brightness_jitter"],
            contrast=corruption["contrast_jitter"],
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=corruption["random_erasing_probability"],
            scale=tuple(corruption["random_erasing_scale"]),
            ratio=tuple(corruption["random_erasing_ratio"]),
            value=0,
        ),
        transforms.ToPILImage(),
    ])


def ingest_data(config: dict) -> dict:
    """
    Main ingestion function. Returns a raw_split_manifest dict recording lineage.

    Algorithm:
      1. For each class, collect all raw images from its synsets.
      2. Shuffle with the global seed.
      3. Allocate the FIRST max_training_images_per_class images → vehicle_dataset (train).
      4. Register their filenames in a global exclusion set.
      5. Allocate the NEXT images_per_class images (non-overlapping) → test_dataset (target).
      6. Apply augmentation to 50% of the test images.
    """
    source_path = Path(config["source_dir"])
    train_path = Path(config["train_dir"])
    test_path = Path(config["test_dir"])
    seed = int(config["seed"])
    max_train = int(config["max_training_images_per_class"])
    n_test = int(config["images_per_class"])
    image_size = tuple(config["image_size"])
    corruption = config["corruption"]

    rng = random.Random(seed)

    # Guard against accidentally merging into existing non-empty directories
    for path, label in [(train_path, "vehicle_dataset"), (test_path, "test_dataset")]:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Refusing to write into non-empty directory '{path}' ({label}). "
                "Delete or move it first, then re-run: dvc repro ingest_data"
            )

    aug_transform = _build_augmentation_pipeline(corruption, image_size)
    clean_transform = transforms.Compose([transforms.Resize(image_size)])

    manifest = {
        "seed": seed,
        "source_dir": str(source_path),
        "splits": {
            "train": {"path": str(train_path), "classes": {}},
            "target": {"path": str(test_path),  "classes": {}},
        },
    }

    for class_idx, (class_name, synsets) in enumerate(CLASS_MAPPING.items()):
        print(f"\nProcessing class: {class_name}")

        all_images = _collect_synset_images(source_path, synsets)
        rng.shuffle(all_images)

        # ── Slice 1: Training images ───────────────────────────────────────────
        train_images = all_images[:max_train]
        # Build exclusion set: raw filenames that went into the training split
        train_filenames = {img.name for img in train_images}

        # ── Slice 2: Test images — skip any raw filename already in training ──
        remaining = [img for img in all_images[max_train:] if img.name not in train_filenames]
        test_images = remaining[:n_test]

        if len(test_images) < n_test:
            print(
                f"  Warning: only {len(test_images)}/{n_test} non-overlapping test images "
                f"found for {class_name}."
            )

        # ── Write training split ───────────────────────────────────────────────
        class_train_dir = train_path / f"{class_idx}_{class_name}"
        class_train_dir.mkdir(parents=True, exist_ok=True)
        train_records = []
        for i, img_path in enumerate(train_images):
            dest = class_train_dir / f"{class_name}_{i:04d}.JPEG"
            shutil.copy2(img_path, dest)
            train_records.append({"src": img_path.name, "dst": str(dest)})
        print(f"  → {len(train_images)} training images written to {class_train_dir}")

        # ── Write test split (50% clean, 50% augmented) ────────────────────────
        class_test_dir = test_path / class_name
        class_test_dir.mkdir(parents=True, exist_ok=True)
        half = len(test_images) // 2
        clean_imgs = test_images[:half]
        aug_imgs = test_images[half:]
        test_records = []

        for i, img_path in enumerate(clean_imgs):
            try:
                img = Image.open(img_path).convert("RGB")
                img = clean_transform(img)
                dest = class_test_dir / f"clean_{i:04d}.jpg"
                img.save(dest)
                test_records.append({"src": img_path.name, "dst": str(dest), "role": "clean"})
            except Exception as exc:
                print(f"  Error (clean) {img_path}: {exc}")

        for i, img_path in enumerate(aug_imgs):
            try:
                img = Image.open(img_path).convert("RGB")
                img = aug_transform(img)
                dest = class_test_dir / f"aug_{i:04d}.jpg"
                img.save(dest)
                test_records.append({"src": img_path.name, "dst": str(dest), "role": "augmented"})
            except Exception as exc:
                print(f"  Error (aug) {img_path}: {exc}")

        print(f"  → {len(clean_imgs)} clean + {len(aug_imgs)} augmented test images written to {class_test_dir}")

        manifest["splits"]["train"]["classes"][class_name] = {
            "count": len(train_records), "images": train_records
        }
        manifest["splits"]["target"]["classes"][class_name] = {
            "count": len(test_records), "images": test_records
        }

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--manifest-out",
        default="data/manifests/raw_split_manifest.json",
        help="Where to write the raw lineage manifest.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    source = Path(config["source_dir"])
    if not source.exists():
        print(
            f"\n[ingest_data] Raw source directory '{source}' not found.\n"
            "This stage requires the raw ImageNet data to be available locally.\n"
            "If you already have pre-built vehicle_dataset/ and test_dataset/, "
            "skip this stage with:\n"
            "  dvc repro --single-item create_data_manifest\n"
        )
        raise SystemExit(1)

    manifest = ingest_data(config)

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Raw split manifest written → {manifest_path}")
    print("   Run 'dvc repro create_data_manifest' to continue the pipeline.")


if __name__ == "__main__":
    main()
