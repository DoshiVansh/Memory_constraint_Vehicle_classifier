try:
    import torch
except ImportError:
    torch = None


def confusion_matrix(targets, predictions, num_classes: int = 5):
    for target, prediction in zip(targets, predictions):
        if not 0 <= int(target) < num_classes or not 0 <= int(prediction) < num_classes:
            raise ValueError("Target and prediction indices must be within the configured class range")
    if torch is not None:
        matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for target, prediction in zip(targets, predictions):
            matrix[int(target), int(prediction)] += 1
        return matrix
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions):
        matrix[int(target)][int(prediction)] += 1
    return matrix


def classification_report(targets, predictions, num_classes: int = 5):
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    matrix = confusion_matrix(targets, predictions, num_classes)
    if torch is not None and isinstance(matrix, torch.Tensor):
        support = [int(v) for v in matrix.sum(dim=1)]
        predicted_support = [int(v) for v in matrix.sum(dim=0)]
        correct = [int(v) for v in matrix.diag()]
        matrix_list = matrix.tolist()
    else:
        support = [sum(matrix[r][c] for c in range(num_classes)) for r in range(num_classes)]
        predicted_support = [sum(matrix[r][c] for r in range(num_classes)) for c in range(num_classes)]
        correct = [matrix[i][i] for i in range(num_classes)]
        matrix_list = matrix

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for index in range(num_classes):
        sup = int(support[index])
        pred_sup = int(predicted_support[index])
        cor = int(correct[index])

        rec = float(cor / sup) if sup > 0 else 0.0
        prec = float(cor / pred_sup) if pred_sup > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)

        per_class[str(index)] = {
            "support": sup,
            "correct": cor,
            "accuracy": rec if sup > 0 else None,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }

    total = len(targets)
    macro_precision = float(sum(precisions) / num_classes) if num_classes > 0 else None
    macro_recall = float(sum(recalls) / num_classes) if num_classes > 0 else None
    macro_f1 = float(sum(f1s) / num_classes) if num_classes > 0 else None

    weighted_precision = (
        float(sum(p * s for p, s in zip(precisions, supports)) / total) if total > 0 else None
    )
    weighted_recall = (
        float(sum(r * s for r, s in zip(recalls, supports)) / total) if total > 0 else None
    )
    weighted_f1 = (
        float(sum(f * s for f, s in zip(f1s, supports)) / total) if total > 0 else None
    )

    return {
        "sample_count": total,
        "accuracy": float(sum(correct) / total) if total else None,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix_list,
    }
