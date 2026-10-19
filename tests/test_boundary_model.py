"""Learned edge geometry, ranking loss, checkpoint reload, and swap checks."""

import random

import pytest
import torch

from jigsaw_ai.baseline_reconstruction import arrangement_cost
from jigsaw_ai.boundary_model import BoundaryCNN, OPPOSITE, boundary_strips, ranking_targets, training_loss
from jigsaw_ai.improve_cnn import cost_dict, fit, raw_costs, refine_swaps, run
from jigsaw_ai.edge_baseline import compatibility_costs
from jigsaw_ai.tensor_data import image_tensor
from jigsaw_ai.solve_boundary import load_boundary_checkpoint, solve
from jigsaw_ai.pairs import DIRECTION_IDS
from jigsaw_ai.generate_puzzles import generate_puzzle, sha256
from jigsaw_ai.plot_cnn_experiments import plot
from jigsaw_ai.evaluate_boundary import evaluate as audit_frozen_boundary
import json
from PIL import Image


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_strips_preserve_edge_orientation_and_classical_costs():
    generator = torch.Generator().manual_seed(42)
    pieces = torch.rand(25, 3, 64, 64, generator=generator)
    strips = boundary_strips(pieces)
    assert torch.equal(strips[:, 0, :, :, 0], pieces[:, :, :, 0])
    assert torch.equal(strips[:, 1, :, :, 0], pieces[:, :, :, -1])
    assert torch.equal(strips[:, 2, :, :, 0], pieces[:, :, 0, :])
    assert torch.equal(strips[:, 3, :, :, 0], pieces[:, :, -1, :])
    images = [Image.new("RGB", (64, 64), (i * 10, i, 250 - i)) for i in range(25)]
    costs = raw_costs(boundary_strips(torch.stack([image_tensor(p) for p in images])))
    classical = compatibility_costs(images)
    for index, direction in enumerate(("LEFT", "RIGHT", "UP", "DOWN")):
        assert float(costs[index, 0, 3]) * 255 == pytest.approx(classical[direction][0][3], abs=1e-5)


@pytest.mark.parametrize("objective", ["random", "hard", "ranking"])
def test_loss_is_finite_backpropagates_and_respects_reciprocity(objective):
    torch.manual_seed(42)
    model = BoundaryCNN()
    strips = boundary_strips(torch.rand(25, 3, 64, 64))
    targets = ranking_targets([divmod(i, 5) for i in range(25)])
    logits = model(strips)
    assert logits.shape == (4, 25, 25)
    for direction, opposite in enumerate(OPPOSITE):
        assert torch.allclose(logits[direction], logits[opposite].T)
    loss = training_loss(logits, targets, objective, torch.Generator().manual_seed(42))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert (targets >= 0).sum() == 80
    assert targets[1, 0] == 1 and targets[3, 0] == 5 and targets[0, 0] == -1


def test_swap_search_never_increases_objective_and_preserves_permutation():
    torch.manual_seed(42)
    tensor = torch.rand(4, 25, 25)
    costs = cost_dict(tensor)
    order = list(range(25))
    random.Random(42).shuffle(order)
    before = arrangement_cost(order, costs)
    result, after = refine_swaps(order, costs)
    assert set(result) == set(range(25)) and len(result) == 25
    assert after <= before
    assert after == pytest.approx(arrangement_cost(result, costs))
    assert refine_swaps(order, costs) == (result, after)
    assert refine_swaps(order, costs, max_rounds=0)[0] == order
    with pytest.raises(ValueError):
        refine_swaps([0] * 25, costs)


def test_short_training_and_checkpoint_reload(tmp_path):
    torch.manual_seed(42)
    positions = [divmod(i, 5) for i in range(25)]
    def record():
        return {"strips": boundary_strips(torch.rand(25, 3, 64, 64)), "targets": ranking_targets(positions)}
    result = fit([record(), record()], [record()], tmp_path / "run", "ranking", 2, {})
    assert result["checkpoint_reload_verified"]
    assert result["optimizer_steps"] == 4
    checkpoint = torch.load(tmp_path / "run/best.pt", weights_only=True)
    model = BoundaryCNN()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    assert checkpoint["architecture"] == "boundary_cnn_v1"
    assert checkpoint["epoch"] == result["best_epoch"]


def test_invalid_geometry_and_objective():
    with pytest.raises(ValueError):
        boundary_strips(torch.zeros(25, 3, 32, 64))
    with pytest.raises(ValueError):
        BoundaryCNN()(torch.zeros(24, 4, 3, 64, 4))
    with pytest.raises(ValueError):
        ranking_targets([(0, 0)] * 25)
    with pytest.raises(ValueError):
        training_loss(torch.zeros(4, 25, 25), torch.zeros(4, 25, dtype=torch.long), "bad", torch.Generator())


def test_standalone_solve_uses_files_without_ground_truth_and_protects_outputs(tmp_path):
    directory = tmp_path / "pieces"
    directory.mkdir()
    for i in range(25):
        Image.new("RGB", (64, 64), (i * 10, 30, 40)).save(directory / f"piece_{i:03}.png")
    checkpoint = tmp_path / "model.pt"
    model = BoundaryCNN()
    torch.save({"architecture": "boundary_cnn_v1", "model_state": model.state_dict(),
                "config": {"direction_ids": DIRECTION_IDS}}, checkpoint)
    first = solve(directory, checkpoint, tmp_path / "first", refine=True)
    second = solve(directory, checkpoint, tmp_path / "second", refine=True)
    assert first == second and len(set(first["order"])) == 25
    with Image.open(tmp_path / "first/reconstruction.png") as image:
        assert image.size == (320, 320)
    assert "accuracy" not in first
    with pytest.raises(ValueError, match="empty"):
        solve(directory, checkpoint, tmp_path / "first")
    with pytest.raises(ValueError, match="overlap"):
        solve(directory, checkpoint, directory / "result")
    with pytest.raises(ValueError, match="0..1"):
        solve(directory, checkpoint, tmp_path / "bad", cnn_weight=2)
    torch.save({"architecture": "siamese_v1"}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="boundary_cnn_v1"):
        load_boundary_checkpoint(tmp_path / "wrong.pt")


def test_experiment_pipeline_does_not_read_test_until_frozen_audit(tmp_path):
    root = tmp_path / "puzzles"
    index = []
    for number, split in enumerate(("train", "validation")):
        image_id = f"{number:016x}"
        source = tmp_path / f"source{number}.png"
        Image.new("RGB", (320, 320), (number * 20, 30, 40)).save(source)
        relative = f"{split}/{image_id}/puzzle.json"
        generate_puzzle(source, (root / relative).parent,
                        {"image_id": image_id, "split": split, "path": source.name, "sha256": sha256(source)}, 42)
        index.append({"image_id": image_id, "split": split, "metadata_path": relative})
    # Index the test puzzle but withhold its files until after model selection.
    test_id = "0000000000000002"
    test_path = f"test/{test_id}/puzzle.json"
    index.append({"image_id": test_id, "split": "test", "metadata_path": test_path})
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    output = tmp_path / "experiments"
    result = run(root, output, random_epochs=1, hard_epochs=1, ranking_epochs=2)
    assert result["test_used"] is False
    assert len(result["training"]) == 3 and len(result["assembly"]) == 10
    assert all(r["checkpoint_reload_verified"] for r in result["training"])
    model, checkpoint = load_boundary_checkpoint(output / result["selected_cnn"] / "best.pt")
    assert not model.training and checkpoint["config"]["test_used"] is False
    plot(output)
    assert (output / "learning_curves.png").exists()
    assert len(list((output / "pure_cnn").glob("*.png"))) == 1
    with pytest.raises(ValueError, match="empty"):
        run(root, output, 1, 1, 1)
    test_source = tmp_path / "test_source.png"
    Image.new("RGB", (320, 320), (40, 30, 40)).save(test_source)
    generate_puzzle(test_source, (root / test_path).parent,
                    {"image_id": test_id, "split": "test", "path": test_source.name, "sha256": sha256(test_source)}, 42)
    audited = audit_frozen_boundary(root, output, tmp_path / "test_audit")
    assert audited["selection_split"] == "validation"
    assert audited["results"]["pure_cnn"]["puzzles"] == 1
    with pytest.raises(ValueError, match="empty"):
        audit_frozen_boundary(root, output, tmp_path / "test_audit")
    with (output / result["selected_cnn"] / "best.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        audit_frozen_boundary(root, output, tmp_path / "changed_audit")
