#!/usr/bin/env python3
"""Create a deterministic data manifest for DVC dataset dependencies."""

import argparse

from vehicle_classifier_core.config import load_config
from vehicle_classifier_core.data_versioning import create_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default="data/manifests/data_manifest.json")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = create_manifest(
        config["datasets"],
        seed=config["seed"],
        validation_fraction=config["validation_fraction"],
    )
    write_json(args.output, manifest)
    print(f"Wrote {len(manifest['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
