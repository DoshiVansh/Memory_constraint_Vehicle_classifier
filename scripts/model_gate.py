#!/usr/bin/env python3
"""Compare a tracked evaluation with the canonical baseline and size limit."""

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vehicle_classifier_core.model_gate import evaluate_model_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="reports/evaluation.json")
    parser.add_argument("--baseline", default="reports/baseline.json")
    parser.add_argument("--config", default="configs/quality_gates.yaml")
    parser.add_argument("--output", default="reports/model_gate.json")
    args = parser.parse_args()
    result = evaluate_model_gate(args.evaluation, args.baseline, args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
