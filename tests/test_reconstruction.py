"""Assembly signals, image-only inference, and held-out evaluation safeguards."""

import json
import random

from PIL import Image
import pytest
import torch

from jigsaw_ai.baseline_metrics import puzzle_metrics
from jigsaw_ai.baseline_reconstruction import greedy_reconstruct, render_arrangement
from jigsaw_ai.checkpoints import save_checkpoint
from jigsaw_ai.evaluate_reconstruction import evaluate
from jigsaw_ai.generate_puzzles import generate_puzzle, sha256
from jigsaw_ai.neural_reconstruction import directional_costs, reconstruct
from jigsaw_ai.pairs import DIRECTIONS, DIRECTION_IDS
from jigsaw_ai.siamese import SiameseCNN


def test_perfect_directional_scores_recover_shuffled_grid():
    permutation = list(range(25))
    random.Random(42).shuffle(permutation)
    positions = [divmod(i, 5) for i in permutation]
    joint = torch.zeros(25, 25, 4)
    for i, (row, column) in enumerate(positions):
        for direction, (dr, dc) in DIRECTIONS.items():
            target = (row + dr, column + dc)
            if target in positions:
                joint[i, positions.index(target), DIRECTION_IDS[direction]] = 1
    before = joint.clone()
    costs = directional_costs({"joint_direction": joint})
    order, cost = greedy_reconstruct(costs)
    assert cost == 0
    assert [positions[i] for i in order] == [divmod(i, 5) for i in range(25)]
    assert puzzle_metrics(order, costs, positions)["exact"]
    assert torch.equal(before, joint)
    assert greedy_reconstruct(costs) == (order, cost)


def test_ties_and_self_scores_cannot_reuse_a_piece():
    joint = torch.full((25, 25, 4), 0.25)
    joint[range(25), range(25), :] = 1
    costs = directional_costs({"joint_direction": joint})
    order, cost = greedy_reconstruct(costs)
    assert order == list(range(25)) and cost == 0.75
    assert all(costs[d][i][i] == float("inf") for d in DIRECTIONS for i in range(25))


@pytest.mark.parametrize("joint", [torch.zeros(24, 25, 4), torch.full((25, 25, 4), float("nan")),
                                   torch.full((25, 25, 4), float("inf")),
                                   torch.full((25, 25, 4), -0.1), torch.full((25, 25, 4), 1.1)])
def test_invalid_scores_rejected(joint):
    with pytest.raises(ValueError):
        directional_costs({"joint_direction": joint})


def test_image_only_api_is_deterministic_and_does_not_change_weights():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = SiameseCNN().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    pieces = [Image.new("RGB", (64, 64), (i * 10, 20, 30)) for i in range(25)]
    first = reconstruct(model, pieces)
    assert first == reconstruct(model, pieces)
    assert sorted(first[0]) == list(range(25))
    assert render_arrangement(pieces, first[0]).size == (320, 320)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_evaluation_outputs_provenance_and_repeatability(tmp_path):
    root = tmp_path / "puzzles"
    index = []
    for number, split in enumerate(("validation", "test")):
        image_id = f"{number:016x}"
        source = tmp_path / f"source{number}.png"
        Image.new("RGB", (320, 320), (20 * number, 30, 40)).save(source)
        record = {"image_id": image_id, "split": split, "path": source.name, "sha256": sha256(source)}
        relative = f"{split}/{image_id}/puzzle.json"
        generate_puzzle(source, (root / relative).parent, record, 42)
        index.append({"image_id": image_id, "split": split, "metadata_path": relative})
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    model = SiameseCNN()
    checkpoint = tmp_path / "best.pt"
    config = {"source_index_sha256": sha256(root / "index.json"), "train_ids": ["separate-training-image"],
              "metadata_sha256": {index[0]["metadata_path"]: sha256(root / index[0]["metadata_path"])}}
    save_checkpoint(checkpoint, model, torch.optim.Adam(model.parameters()), 1, config, {})
    validation, test = tmp_path / "validation", tmp_path / "test"
    with pytest.raises(ValueError, match="completed validation"):
        evaluate(root, checkpoint, test, "test")
    evaluate(root, checkpoint, validation, "validation")
    result = evaluate(root, checkpoint, test, "test", validation)
    records = json.loads((test / "results.json").read_text())
    assert result["puzzles"] == 1
    assert len(records[0]["order"]) == len(set(records[0]["order"])) == 25
    assert result["fragment_position_accuracy"] == records[0]["position_correct"] / 25
    assert result["assembly_neighbor_accuracy"] == records[0]["assembly_borders_correct"] / 40
    with Image.open(test / "predictions" / f"{index[1]['image_id']}.png") as image:
        assert image.size == (320, 320)
    hashes = {str(p.relative_to(test)): sha256(p) for p in test.rglob("*") if p.is_file()}
    evaluate(root, checkpoint, test, "test", validation)
    assert hashes == {str(p.relative_to(test)): sha256(p) for p in test.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overlap"):
        evaluate(root, checkpoint, root / "output", "validation")
    with pytest.raises(ValueError, match="contain the checkpoint"):
        evaluate(root, checkpoint, tmp_path, "validation")
    config["train_ids"].append(index[1]["image_id"])
    save_checkpoint(tmp_path / "overlapping.pt", model, torch.optim.Adam(model.parameters()), 1, config, {})
    with pytest.raises(ValueError, match="overlap training"):
        evaluate(root, tmp_path / "overlapping.pt", tmp_path / "bad", "test", validation)
    changed = json.loads((validation / "config.json").read_text())
    changed["method"]["version"] = 999
    (validation / "config.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="frozen method"):
        evaluate(root, checkpoint, tmp_path / "mismatch", "test", validation)
    with pytest.raises(ValueError, match="different inputs"):
        evaluate(root, checkpoint, validation, "validation")
