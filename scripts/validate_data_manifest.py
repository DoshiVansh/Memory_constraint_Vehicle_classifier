#!/usr/bin/env python3
"""Validate image readability, class coverage, duplicates, and split leakage."""

import argparse
import json
import sys

from vehicle_classifier_core.data_versioning import validate_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/data_manifest.json")
    parser.add_argument("--output", default="reports/data_validation.json")
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)
    report = validate_manifest(manifest, near_duplicate_distance=args.near_duplicate_distance)
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "errors": len(report["errors"]), "warnings": len(report["warnings"])}))
    if report["status"] == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
