"""Metrics calculated from predictions and ground truth, separate from solving."""

from itertools import groupby

OFFSETS = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (-1, 0), "DOWN": (1, 0)}


def pair_predictions(costs: dict, positions: list[tuple[int, int]]) -> list[tuple[float, int]]:
    result = []
    for i, (row, column) in enumerate(positions):
        for j, (other_row, other_column) in enumerate(positions):
            if i != j:
                cost = min(costs[d][i][j] for d in OFFSETS)
                label = int(abs(row - other_row) + abs(column - other_column) == 1)
                result.append((cost, label))
    return result


def classification_metrics(predictions: list[tuple[float, int]], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for cost, label in predictions:
        if cost <= threshold:
            tp += label
            fp += 1 - label
        else:
            fn += label
            tn += 1 - label
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / len(predictions) if predictions else 0.0}


def select_threshold(predictions: list[tuple[float, int]]) -> float:
    """Maximize validation F1 only; group equal costs to avoid impossible cutoffs."""
    if not predictions or not 0 < sum(label for _, label in predictions) < len(predictions):
        raise ValueError("Threshold selection requires both positive and negative examples")
    remaining = sum(label for _, label in predictions)
    tp = fp = 0
    best_f1, best_threshold = -1.0, -1.0
    for cost, group in groupby(sorted(predictions), key=lambda pair: pair[0]):
        labels = [label for _, label in group]
        tp += sum(labels)
        fp += len(labels) - sum(labels)
        remaining -= sum(labels)
        f1 = 2 * tp / (2 * tp + fp + remaining)
        if f1 > best_f1:
            best_f1, best_threshold = f1, cost
    return best_threshold


def puzzle_metrics(order: list[int], costs: dict, positions: list[tuple[int, int]]) -> dict:
    if len(order) != 25 or set(order) != set(range(25)):
        raise ValueError("Expected a complete arrangement")
    if len(positions) != 25 or set(positions) != {(r, c) for r in range(5) for c in range(5)}:
        raise ValueError("Expected complete 5x5 ground truth")
    lookup = {position: i for i, position in enumerate(positions)}
    neighbor_correct = 0
    for i, (row, column) in enumerate(positions):
        for direction, (dr, dc) in OFFSETS.items():
            target = lookup.get((row + dr, column + dc))
            if target is not None:
                predicted = min((j for j in range(25) if j != i), key=lambda j: (costs[direction][i][j], j))
                neighbor_correct += int(predicted == target)
    position_correct = sum(positions[piece] == divmod(index, 5) for index, piece in enumerate(order))
    borders_correct = 0
    for index, piece in enumerate(order):
        row, column = divmod(index, 5)
        a = positions[piece]
        if column < 4:
            b = positions[order[index + 1]]
            borders_correct += int(b == (a[0], a[1] + 1))
        if row < 4:
            b = positions[order[index + 5]]
            borders_correct += int(b == (a[0] + 1, a[1]))
    return {"neighbor_correct": neighbor_correct, "neighbor_queries": 80,
            "neighbor_prediction_accuracy": neighbor_correct / 80,
            "position_correct": position_correct, "position_accuracy": position_correct / 25,
            "assembly_borders_correct": borders_correct, "assembly_neighbor_accuracy": borders_correct / 40,
            "exact": position_correct == 25}
