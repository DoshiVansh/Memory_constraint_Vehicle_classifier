#!/usr/bin/env python3
"""Create the checksum and contract metadata required by the inference API."""

import argparse
import json
from pathlib import Path

from vehicle_classifier_core.artifact_validation import ARCHITECTURE
from vehicle_classifier_core.model import CLASS_IDX
from vehicle_classifier_core.preprocessing import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from vehicle_classifier_core.reproducibility import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    metadata = {
        "architecture": ARCHITECTURE,
        "class_mapping": {str(index): label for index, label in CLASS_IDX.items()},
        "preprocessing": {
            "color": "RGB",
            "size": list(IMAGE_SIZE),
            "mean": list(NORMALIZE_MEAN),
            "std": list(NORMALIZE_STD),
        },
        "checkpoint": args.checkpoint.name,
        "sha256": file_sha256(args.checkpoint),
        "size_bytes": args.checkpoint.stat().st_size,
    }
    output = args.output or Path(f"{args.checkpoint}.json")
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output} for {args.checkpoint}")


if __name__ == "__main__":
    main()
