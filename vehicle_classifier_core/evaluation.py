"""Evaluation metrics independent of console formatting or log parsing."""

import torch


def confusion_matrix(targets, predictions, num_classes: int = 5):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets, predictions):
        if not 0 <= int(target) < num_classes or not 0 <= int(prediction) < num_classes:
            raise ValueError("Target and prediction indices must be within the configured class range")
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_report(targets, predictions, num_classes: int = 5):
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    matrix = confusion_matrix(targets, predictions, num_classes)
    support = matrix.sum(dim=1)
    correct = matrix.diag()
    per_class = {}
    for index in range(num_classes):
        per_class[str(index)] = {
            "support": int(support[index]),
            "correct": int(correct[index]),
            "accuracy": float(correct[index] / support[index]) if support[index] else None,
        }
    total = len(targets)
    return {
        "sample_count": total,
        "accuracy": float(correct.sum() / total) if total else None,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
