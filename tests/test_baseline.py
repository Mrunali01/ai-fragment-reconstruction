"""Controlled edge signals and hand-calculated metrics verify the baseline."""

import json
import random

from PIL import Image
import pytest

from jigsaw_ai.edge_baseline import compatibility_costs, edge_difference
from jigsaw_ai.baseline_reconstruction import arrangement_cost, greedy_reconstruct, render_arrangement
from jigsaw_ai.baseline_metrics import classification_metrics, pair_predictions, puzzle_metrics, select_threshold
from jigsaw_ai.evaluate_baseline import evaluate
from jigsaw_ai.generate_puzzles import generate_puzzle, sha256


def coded_pieces():
    def color(n):
        return (n * 3 + 1, n * 4 + 2, n * 2 + 3)
    pieces = []
    for row in range(5):
        for column in range(5):
            piece = Image.new("RGB", (64, 64), "white")
            for offset in range(1, 63):
                piece.putpixel((0, offset), color(row * 6 + column))
                piece.putpixel((63, offset), color(row * 6 + column + 1))
                piece.putpixel((offset, 0), color(30 + row * 5 + column))
                piece.putpixel((offset, 63), color(30 + (row + 1) * 5 + column))
            for corner in ((0, 0), (0, 63), (63, 0), (63, 63)):
                piece.putpixel(corner, (0, 0, 0))
            pieces.append(piece)
    permutation = list(range(25))
    random.Random(42).shuffle(permutation)
    return [pieces[i] for i in permutation], [divmod(i, 5) for i in permutation]


def test_rgb_mean_difference_and_opposite_symmetry():
    a = Image.new("RGB", (64, 64), (0, 10, 20))
    b = Image.new("RGB", (64, 64), (30, 40, 80))
    assert edge_difference(a, b, "RIGHT") == 40
    assert edge_difference(a, b, "UP") == edge_difference(b, a, "DOWN")
    with pytest.raises(ValueError, match="Unknown direction"):
        edge_difference(a, b, "DIAGONAL")
    with pytest.raises(ValueError, match="64x64"):
        edge_difference(a.resize((32, 32)), b, "LEFT")


def test_solver_recovers_controlled_puzzle_without_labels():
    pieces, positions = coded_pieces()
    costs = compatibility_costs(pieces)
    assert all(costs["RIGHT"][i][i] == float("inf") for i in range(25))
    order, cost = greedy_reconstruct(costs)
    assert cost == 0
    assert [positions[i] for i in order] == [divmod(i, 5) for i in range(25)]
    metrics = puzzle_metrics(order, costs, positions)
    assert metrics["neighbor_prediction_accuracy"] == 1
    assert metrics["position_accuracy"] == 1
    assert metrics["assembly_neighbor_accuracy"] == 1
    assert metrics["exact"] is True
    pairs = pair_predictions(costs, positions)
    assert len(pairs) == 600 and sum(label for _, label in pairs) == 80
    assert select_threshold(pairs) == 0
    assert classification_metrics(pairs, 0)["f1"] == 1
    assert render_arrangement(pieces, order).size == (320, 320)
    assert greedy_reconstruct(costs) == (order, cost)


def test_ties_are_deterministic_and_no_piece_is_reused():
    pieces = [Image.new("RGB", (64, 64), "white") for _ in range(25)]
    costs = compatibility_costs(pieces)
    order, cost = greedy_reconstruct(costs)
    assert order == list(range(25)) and cost == 0
    with pytest.raises(ValueError, match="exactly once"):
        arrangement_cost([0] * 25, costs)
    with pytest.raises(ValueError, match="25 pieces"):
        render_arrangement(pieces, order[:-1])


def test_threshold_ties_and_hand_calculated_confusion_matrix():
    predictions = [(1, 1), (1, 0), (2, 1), (3, 0)]
    result = classification_metrics(predictions, 2)
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (2, 1, 1, 0)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1 and result["f1"] == 0.8
    assert select_threshold(predictions) == 2
    assert select_threshold([(1, 1), (1, 0)]) == 1
    with pytest.raises(ValueError, match="both positive and negative"):
        select_threshold([(1, 1)])


def test_position_metrics_do_not_allow_global_translation():
    pieces, positions = coded_pieces()
    costs = compatibility_costs(pieces)
    lookup = {p: i for i, p in enumerate(positions)}
    shifted = [lookup[r, (c + 1) % 5] for r in range(5) for c in range(5)]
    metrics = puzzle_metrics(shifted, costs, positions)
    assert metrics["position_accuracy"] == 0
    assert metrics["exact"] is False
    assert 0 < metrics["assembly_neighbor_accuracy"] < 1


def test_validation_calibration_required_for_test_and_reproducible(tmp_path):
    root = tmp_path / "puzzles"
    index = []
    for i, split in enumerate(("validation", "test")):
        image_id = f"{i:016x}"
        source = tmp_path / f"source_{i}.png"
        Image.new("RGB", (320, 320), "white").save(source)
        record = {"image_id": image_id, "split": split, "path": source.name, "sha256": sha256(source)}
        relative = f"{split}/{image_id}/puzzle.json"
        generate_puzzle(source, (root / relative).parent, record, 42)
        index.append({"image_id": image_id, "split": split, "metadata_path": relative})
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="requires validation"):
        evaluate(root, tmp_path / "test", "test")
    validation = tmp_path / "validation"
    val = evaluate(root, validation, "validation")
    test = tmp_path / "test"
    result = evaluate(root, test, "test", validation / "calibration.json")
    assert result["threshold"] == val["threshold"]
    assert result["pair_examples"] == 600
    before = {str(p.relative_to(test)): sha256(p) for p in test.rglob("*") if p.is_file()}
    evaluate(root, test, "test", validation / "calibration.json")
    assert before == {str(p.relative_to(test)): sha256(p) for p in test.rglob("*") if p.is_file()}
    calibration = json.loads((validation / "calibration.json").read_text())
    calibration["split"] = "test"
    (validation / "calibration.json").write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="validation split"):
        evaluate(root, tmp_path / "other_test", "test", validation / "calibration.json")
