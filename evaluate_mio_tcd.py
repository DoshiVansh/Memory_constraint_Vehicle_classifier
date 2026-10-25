"""Evaluate the mapped MIO-TCD stress-test data with the shared evaluator."""

import argparse
import logging

from evaluate_test_set import evaluate
from vehicle_classifier_core.reproducibility import configure_logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ood_evaluation.yaml")
    args = parser.parse_args()
    configure_logging()
    try:
        evaluate(args.config)
    except Exception:
        logging.getLogger(__name__).exception("OOD evaluation failed", extra={"config": args.config})
        raise
