"""Training, checkpoint provenance, cached inference, and data separation checks."""

import json
import math

from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from jigsaw_ai.checkpoints import load_checkpoint
from jigsaw_ai.evaluate_model import evaluate_model
from jigsaw_ai.generate_puzzles import generate_puzzle, sha256
from jigsaw_ai.model_inference import neighbor_counts, score_puzzle
from jigsaw_ai.siamese import SiameseCNN
from jigsaw_ai.tensor_data import collate_pairs, image_tensor
from jigsaw_ai.train import TrainingConfig, run_epoch, train


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def root(tmp_path):
    directory = tmp_path / "puzzles"
    index = []
    for number, split in enumerate(("train", "validation", "test")):
        image_id = f"{number:016x}"
        source = tmp_path / f"source_{number}.png"
        image = Image.new("RGB", (320, 320))
        image.putdata([((x + 17 * number) % 256, y % 256, (x * 3 + y) % 256)
                       for y in range(320) for x in range(320)])
        image.save(source)
        record = {"image_id": image_id, "split": split, "path": source.name, "sha256": sha256(source)}
        relative = f"{split}/{image_id}/puzzle.json"
        generate_puzzle(source, (directory / relative).parent, record, 42)
        index.append({"image_id": image_id, "split": split, "metadata_path": relative})
    (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return directory


def memory_samples():
    white, black = Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "black")
    return [{"first": white, "second": black, "compatible": 1, "direction": 0},
            {"first": black, "second": white, "compatible": 0, "direction": None}]


def test_tiny_subset_can_be_learned_and_validation_does_not_update_weights():
    torch.manual_seed(42)
    model = SiameseCNN()
    loader = DataLoader(memory_samples(), batch_size=2, collate_fn=collate_pairs)
    initial = run_epoch(model, loader, 1)["loss"]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(35):
        run_epoch(model, loader, 1, optimizer)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    final = run_epoch(model, loader, 1)["loss"]
    assert final < initial * 0.5
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_epoch_loss_aggregation_uses_positive_count():
    model = SiameseCNN()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    samples = [memory_samples()[0]] * 3 + [memory_samples()[1]]
    stats = run_epoch(model, DataLoader(samples, batch_size=3, collate_fn=collate_pairs), 1)
    assert stats["examples"] == 4 and stats["positive_examples"] == 3
    assert stats["compatibility_loss"] == pytest.approx(math.log(2))
    assert stats["direction_loss"] == pytest.approx(math.log(4))


def test_cached_inference_matches_direct_forward():
    torch.manual_seed(42)
    model = SiameseCNN().eval()
    pieces = [Image.new("RGB", (64, 64), (i * 9, 255 - i * 9, i * 5)) for i in range(25)]
    scores = score_puzzle(model, pieces, batch_size=101)
    assert scores["compatibility"].shape == (25, 25)
    assert scores["joint_direction"].shape == (25, 25, 4)
    with torch.inference_mode():
        outputs = model(torch.stack([image_tensor(pieces[i]) for i in (0, 4, 24)]),
                        torch.stack([image_tensor(pieces[j]) for j in (3, 0, 2)]))
    for k, (i, j) in enumerate(((0, 3), (4, 0), (24, 2))):
        assert scores["compatibility"][i, j] == pytest.approx(outputs["compatibility_logits"][k].sigmoid().item(), abs=1e-6)
        assert torch.allclose(scores["conditional_direction"][i, j], outputs["direction_logits"][k].softmax(0), atol=1e-6)
    assert scores["compatibility"].diagonal().sum() == 0
    with pytest.raises(ValueError, match="model.eval"):
        score_puzzle(model.train(), pieces)


def test_neighbor_ranking_uses_all_candidates_and_excludes_self():
    positions = [divmod(i, 5) for i in range(25)]
    scores = {"joint_direction": torch.zeros(25, 25, 4)}
    for i, (row, column) in enumerate(positions):
        for direction, (dr, dc) in enumerate(((0, -1), (0, 1), (-1, 0), (1, 0))):
            target = (row + dr, column + dc)
            scores["joint_direction"][i, i, direction] = 100
            if target in positions:
                scores["joint_direction"][i, positions.index(target), direction] = 1
    assert neighbor_counts(scores, positions) == (80, 80)


def test_training_checkpoint_reload_and_evaluation_provenance(root, tmp_path):
    output = tmp_path / "training"
    settings = TrainingConfig(epochs=2, batch_size=80, max_seconds=60)
    summary = train(root, output, settings)
    assert summary["weights_changed"] and summary["optimizer_steps"] == 4
    assert summary["checkpoint_reload_verified"]
    history = json.loads((output / "history.json").read_text())
    assert summary["best_validation"]["loss"] == min(r["validation"]["loss"] for r in history)
    assert (output / "learning_curves.png").exists()
    model, checkpoint = load_checkpoint(output / "best.pt")
    assert not model.training and checkpoint["optimizer_state"]["state"]
    assert not set(checkpoint["config"]["train_ids"]) & set(checkpoint["config"]["validation_ids"])
    validation = tmp_path / "eval_validation"
    val = evaluate_model(root, output / "best.pt", validation, "validation")
    test = evaluate_model(root, output / "best.pt", tmp_path / "eval_test", "test", validation / "calibration.json")
    assert test["pair_examples"] == 600 and test["neighbor_queries"] == 80
    assert test["compatibility_probability_threshold"] == val["compatibility_probability_threshold"]
    assert "fragment_position_accuracy" not in test
    with pytest.raises(ValueError, match="requires validation"):
        evaluate_model(root, output / "best.pt", tmp_path / "missing_calibration", "test")
    with pytest.raises(ValueError, match="must be empty"):
        train(root, output, settings)
    calibration = json.loads((validation / "calibration.json").read_text())
    calibration["checkpoint_sha256"] = "wrong-checkpoint"
    (validation / "calibration.json").write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="Calibration must match"):
        evaluate_model(root, output / "best.pt", tmp_path / "bad_calibration", "test", validation / "calibration.json")
    checkpoint["config"]["train_ids"].append("0000000000000001")
    torch.save(checkpoint, tmp_path / "overlapping.pt")
    with pytest.raises(ValueError, match="overlap training"):
        evaluate_model(root, tmp_path / "overlapping.pt", tmp_path / "overlapping_evaluation", "validation")


def test_early_stopping_keeps_first_best_checkpoint(root, tmp_path):
    result = train(root, tmp_path / "early_stop", TrainingConfig(
        epochs=4, batch_size=80, learning_rate=1e-30, patience=1, max_seconds=60))
    assert result["stop_reason"] == "early_stopping"
    assert result["epochs_recorded"] == 2 and result["best_epoch"] == 1


def test_seeded_training_reproduces_weights(root, tmp_path):
    settings = TrainingConfig(epochs=1, batch_size=80, max_seconds=60)
    first = train(root, tmp_path / "repeat_a", settings)
    second = train(root, tmp_path / "repeat_b", settings)
    a, _ = load_checkpoint(tmp_path / "repeat_a/best.pt")
    b, _ = load_checkpoint(tmp_path / "repeat_b/best.pt")
    assert first["best_validation"] == second["best_validation"]
    assert all(torch.equal(a.state_dict()[key], value) for key, value in b.state_dict().items())


def test_time_budget_records_partial_epoch(root, tmp_path):
    result = train(root, tmp_path / "bounded", TrainingConfig(epochs=3, batch_size=32, max_seconds=1e-12))
    assert result["stop_reason"] == "time_budget"
    assert result["optimizer_steps"] == 1 and result["complete_epochs"] == 0
    assert result["checkpoint_reload_verified"]


@pytest.mark.parametrize("settings", [TrainingConfig(epochs=0), TrainingConfig(learning_rate=0),
                                      TrainingConfig(direction_weight=-1), TrainingConfig(train_puzzles=0)])
def test_bad_training_settings_rejected(settings):
    with pytest.raises(ValueError):
        settings.validate()
