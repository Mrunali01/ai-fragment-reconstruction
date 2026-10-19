"""Group moves, deterministic budgets, and validation/fresh-cohort safeguards."""

import json

import numpy as np
from PIL import Image
import pytest
import torch

from jigsaw_ai.baseline_reconstruction import arrangement_cost
from jigsaw_ai.boundary_model import BoundaryCNN
from jigsaw_ai.fresh_holdout import difference_hash
from jigsaw_ai.generate_puzzles import generate_puzzle, sha256
from jigsaw_ai.group_search import candidates, group_refine, objective
from jigsaw_ai.improve_cnn import cost_dict, refine_swaps
from jigsaw_ai.pairs import DIRECTION_IDS
from jigsaw_ai.search_experiments import evaluate


def test_group_moves_repair_a_cycle_that_single_swaps_cannot():
    tensor = torch.ones(4, 25, 25)
    for i in range(25):
        row, column = divmod(i, 5)
        if column < 4:
            tensor[1, i, i + 1] = tensor[0, i + 1, i] = 0
        if row < 4:
            tensor[3, i, i + 5] = tensor[2, i + 5, i] = 0
    costs = cost_dict(tensor)
    order = list(range(20)) + [21, 22, 23, 24, 20]
    assert refine_swaps(order, costs)[0] == order
    repaired, cost = group_refine(order, costs)
    assert repaired == list(range(25)) and cost == 0
    assert group_refine(order, costs, "beam") == (repaired, cost)


def test_candidate_permutations_and_vectorized_cost_match_scalar():
    order = list(range(25))
    moves = candidates(order)
    assert len(moves) > 400
    assert all(sorted(row) == order for row in moves.tolist())
    assert len(np.unique(moves, axis=0)) == len(moves)
    costs = cost_dict(torch.rand(4, 25, 25, generator=torch.Generator().manual_seed(42)))
    scores = objective(moves, costs)
    for index in (0, len(moves) // 2, len(moves) - 1):
        assert scores[index] == pytest.approx(arrangement_cost(moves[index].tolist(), costs))
    for mode in ("groups", "beam"):
        result, score = group_refine(order, costs, mode)
        assert score <= arrangement_cost(order, costs)
        assert sorted(result) == order
        assert group_refine(order, costs, mode) == (result, score)
    with pytest.raises(ValueError):
        group_refine([0] * 25, costs)


def test_difference_hash_is_stable_and_detects_identical_prepared_images():
    image = Image.new("RGB", (320, 320), "white")
    assert difference_hash(image) == difference_hash(image.copy())
    ramp = Image.fromarray(np.tile(np.arange(255, -1, -1, dtype=np.uint8), (256, 1)))
    assert (difference_hash(image) ^ difference_hash(ramp)).bit_count() > 4


def test_frozen_evaluation_rejects_reused_development_test_images(tmp_path):
    torch.set_num_threads(2)
    root = tmp_path / "puzzles"
    records = []
    for number, split in enumerate(("train", "validation", "test")):
        image_id = f"{number:016x}"
        source = tmp_path / f"image{number}.png"
        Image.new("RGB", (320, 320), (10 * number, 20, 30)).save(source)
        relative = f"{split}/{image_id}/puzzle.json"
        generate_puzzle(source, (root / relative).parent,
                        {"image_id": image_id, "split": split, "path": source.name, "sha256": sha256(source)}, 42)
        records.append({"image_id": image_id, "split": split, "metadata_path": relative})
    (root / "index.json").write_text(json.dumps(records))
    model = BoundaryCNN()
    checkpoint = tmp_path / "model.pt"
    payload = {"architecture": "boundary_cnn_v1", "model_state": model.state_dict(),
               "config": {"direction_ids": DIRECTION_IDS, "train_ids": [records[0]["image_id"]],
                          "source_index_sha256": sha256(root / "index.json"),
                          "metadata_sha256": {records[1]["metadata_path"]: sha256(root / records[1]["metadata_path"])}}}
    torch.save(payload, checkpoint)
    validation = tmp_path / "validation"
    report = evaluate(root, checkpoint, validation)
    assert report["selected_mode"] == "swaps"  # Exact ties retain the incumbent.
    with pytest.raises(ValueError, match="overlaps development"):
        evaluate(root, checkpoint, tmp_path / "reused", validation)
    fresh = tmp_path / "fresh"
    image_id = "0000000000000003"
    relative = f"test/{image_id}/puzzle.json"
    source = tmp_path / "fresh.png"
    Image.new("RGB", (320, 320), (30, 20, 30)).save(source)
    generate_puzzle(source, (fresh / relative).parent,
                    {"image_id": image_id, "split": "test", "path": source.name, "sha256": sha256(source)}, 42)
    (fresh / "index.json").write_text(json.dumps([{"image_id": image_id, "split": "test", "metadata_path": relative}]))
    audited = evaluate(fresh, checkpoint, tmp_path / "fresh_audit", validation)
    assert audited["cnn_retrained"] is False and audited["selected_mode"] == "swaps"
    assert audited["reports"][0]["puzzles"] == 1
    with pytest.raises(ValueError, match="empty"):
        evaluate(fresh, checkpoint, tmp_path / "fresh_audit", validation)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"validation_selection_sha256": sha256(validation / "summary.json"),
                                    "checkpoint_sha256": sha256(checkpoint), "selected_search": "swaps", "count": 99}))
    with pytest.raises(ValueError, match="protocol"):
        evaluate(fresh, checkpoint, tmp_path / "wrong_protocol", validation, protocol)
