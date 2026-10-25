"""Version-neutral manifest creation and validation for classifier datasets."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CANONICAL_CLASSES = ("Bus", "Truck", "Car", "Bike", "None")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def split_indices(sample_count: int, seed: int, validation_fraction: float, group_ids=None):
    """Return deterministic validation/train indices, keeping groups intact."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    target_validation_count = int(sample_count * validation_fraction)
    if target_validation_count < 1 or target_validation_count >= sample_count:
        raise ValueError("Dataset is too small for the requested validation fraction")
    if group_ids is None:
        group_ids = list(range(sample_count))
    if len(group_ids) != sample_count:
        raise ValueError("group_ids must have one entry per sample")
    groups = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    grouped_indices = list(groups.values())
    random.Random(seed).shuffle(grouped_indices)
    validation_indices = []
    train_indices = []
    for group in grouped_indices:
        if len(validation_indices) < target_validation_count:
            validation_indices.extend(group)
        else:
            train_indices.extend(group)
    return sorted(validation_indices), sorted(train_indices)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    """Return a 64-bit dHash used only to surface possible near duplicates."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{bits:016x}"


def _class_directories(dataset_path: Path, prefixed: bool):
    expected = [f"{index}_{name}" for index, name in enumerate(CANONICAL_CLASSES)] if prefixed else list(CANONICAL_CLASSES)
    actual = sorted(path.name for path in dataset_path.iterdir() if path.is_dir())
    if actual != sorted(expected):
        raise ValueError(f"{dataset_path}: expected class directories {sorted(expected)}, found {actual}")
    return [(index, name, dataset_path / (f"{index}_{name}" if prefixed else name)) for index, name in enumerate(CANONICAL_CLASSES)]


def _assign_split(role: str, relative_path: str, index: int, *, seed: int, validation_fraction: float):
    if role == "training":
        # Stable per-class shuffle prevents class-order bias and makes the split reproducible.
        return None
    filename = Path(relative_path).name.lower()
    if filename.startswith("clean_"):
        return "test_clean" if role == "target" else "ood_clean"
    if filename.startswith("aug_"):
        return "test_corrupted" if role == "target" else "ood_corrupted"
    return "test" if role == "target" else "ood"


def create_manifest(dataset_specs: dict, *, seed: int, validation_fraction: float) -> dict:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    records = []
    summaries = {}
    for dataset_name in ("training", "target", "ood"):
        spec = dataset_specs[dataset_name]
        root = Path(spec["path"])
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        role = spec["role"]
        if role != dataset_name:
            raise ValueError(f"Dataset {dataset_name} must declare role={dataset_name!r}")
        class_dirs = _class_directories(root, bool(spec.get("prefixed_classes", False)))
        dataset_records = []
        class_counts = Counter()
        for class_index, class_name, class_dir in class_dirs:
            files = sorted(
                (path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                key=lambda path: path.relative_to(root).as_posix(),
            )
            class_counts[class_name] = len(files)
            for path in files:
                relative_path = path.relative_to(root).as_posix()
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        width, height = image.size
                        perceptual_hash = difference_hash(image)
                except Exception as exc:
                    width = height = None
                    perceptual_hash = None
                    decode_error = f"{type(exc).__name__}: {exc}"
                else:
                    decode_error = None
                dataset_records.append(
                    {
                        "dataset": dataset_name,
                        "role": role,
                        "path": relative_path,
                        "source": spec.get("source", "unspecified"),
                        "class_name": class_name,
                        "class_index": class_index,
                        "split": _assign_split(role, relative_path, class_index, seed=seed, validation_fraction=validation_fraction),
                        "width": width,
                        "height": height,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "dhash64": perceptual_hash,
                        "decode_error": decode_error,
                    }
                )
        records.extend(dataset_records)
        summaries[dataset_name] = {
            "path": spec.get("manifest_path", Path(spec["path"]).name),
            "source": spec.get("source", "unspecified"),
            "samples": len(dataset_records),
            "class_counts": dict(sorted(class_counts.items())),
        }
    train_records = [row for row in records if row["dataset"] == "training"]
    validation_indices, _ = split_indices(
        len(train_records), seed, validation_fraction, [row["sha256"] for row in train_records]
    )
    validation_indices = set(validation_indices)
    for index, row in enumerate(train_records):
        row["split"] = "validation" if index in validation_indices else "train"
    records.sort(key=lambda row: (row["dataset"], row["class_index"], row["split"], row["path"]))
    return {
        "schema_version": 1,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "training_split_rule": "Shuffle exact-SHA256 groups with random.Random(seed); assign whole groups to validation until the target fraction is reached",
        "perceptual_hash": "dhash64-v1",
        "class_mapping": {str(index): name for index, name in enumerate(CANONICAL_CLASSES)},
        "datasets": summaries,
        "records": records,
    }


class _BKTree:
    def __init__(self):
        self.root = None

    @staticmethod
    def _distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, record: dict):
        node = [value, [record], {}]
        if self.root is None:
            self.root = node
            return
        current = self.root
        while True:
            distance = self._distance(value, current[0])
            if distance == 0:
                current[1].append(record)
                return
            if distance not in current[2]:
                current[2][distance] = node
                return
            current = current[2][distance]

    def query(self, value: int, radius: int):
        if self.root is None:
            return []
        found = []
        pending = [self.root]
        while pending:
            current = pending.pop()
            distance = self._distance(value, current[0])
            if distance <= radius:
                found.extend((record, distance) for record in current[1])
            pending.extend(
                child for edge_distance, child in current[2].items()
                if distance - radius <= edge_distance <= distance + radius
            )
        return found


def validate_manifest(manifest: dict, *, near_duplicate_distance: int = 4) -> dict:
    errors = []
    warnings = []
    counts = Counter()
    class_presence = defaultdict(set)
    exact = defaultdict(list)
    training_hashes = defaultdict(_BKTree)
    for row in manifest["records"]:
        counts[(row["dataset"], row["split"])] += 1
        class_presence[row["dataset"]].add(row["class_name"])
        if row["decode_error"] or row["width"] is None or row["height"] is None:
            errors.append({"type": "unreadable_image", "dataset": row["dataset"], "path": row["path"], "detail": row["decode_error"]})
        if row["size_bytes"] <= 0:
            errors.append({"type": "empty_file", "dataset": row["dataset"], "path": row["path"]})
        exact[row["sha256"]].append(row)
        if row["dhash64"] and row["split"] == "train":
            training_hashes[row["class_name"]].add(int(row["dhash64"], 16), row)
    for dataset_name in manifest["datasets"]:
        missing = sorted(set(CANONICAL_CLASSES) - class_presence[dataset_name])
        if missing:
            errors.append({"type": "missing_classes", "dataset": dataset_name, "classes": missing})
    for digest, rows in exact.items():
        if len(rows) > 1:
            split_pairs = sorted({row["split"] for row in rows})
            duplicate = {
                "sha256": digest,
                "locations": [f"{row['dataset']}:{row['path']}" for row in rows],
                "splits": split_pairs,
            }
            if len(split_pairs) > 1:
                errors.append({"type": "duplicate_across_splits", **duplicate})
            else:
                warnings.append({"type": "duplicate_within_split", **duplicate})
    near_seen = set()
    near_duplicate_count = 0
    max_warning_examples = 250
    warnings_truncated = False
    for row in manifest["records"]:
        if warnings_truncated:
            break
        if row["split"] == "train" or not row["dhash64"]:
            continue
        tree = training_hashes[row["class_name"]]
        for training_row, distance in tree.query(int(row["dhash64"], 16), near_duplicate_distance):
            pair = tuple(sorted((f"{training_row['dataset']}:{training_row['path']}", f"{row['dataset']}:{row['path']}")))
            if pair in near_seen or training_row["sha256"] == row["sha256"]:
                continue
            near_seen.add(pair)
            near_duplicate_count += 1
            if len(warnings) < max_warning_examples:
                warnings.append({"type": "possible_near_duplicate", "distance": distance, "locations": list(pair)})
            if near_duplicate_count >= max_warning_examples:
                warnings_truncated = True
                break
    report = {
        "schema_version": 1,
        "status": "failed" if errors else ("review_warnings" if warnings else "passed"),
        "sample_counts": {
            dataset: {split: amount for (name, split), amount in sorted(counts.items()) if name == dataset}
            for dataset in manifest["datasets"]
        },
        "errors": errors,
        "warnings": warnings,
        "near_duplicate_candidate_count": near_duplicate_count,
        "warnings_truncated": warnings_truncated,
        "warning_note": "Perceptual-hash matches are review candidates, not proof of leakage.",
    }
    return report


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
