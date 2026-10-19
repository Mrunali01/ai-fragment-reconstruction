"""Image-only reconstruction, independent metric checks, and report provenance."""

import json

import numpy as np
from PIL import Image
import pytest
import torch

from jigsaw_ai.boundary_model import BoundaryCNN
from jigsaw_ai.evaluate_pipeline import run
from jigsaw_ai.evaluation_report import aggregate, appearance_metrics, paired_comparison, wilson_interval
from jigsaw_ai.generate_puzzles import generate_puzzle, sha256
from jigsaw_ai.pairs import DIRECTION_IDS
from jigsaw_ai.pipeline import reconstruct, score_pieces
from jigsaw_ai.search_experiments import METHOD as SEARCH_METHOD


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_paired_statistics_use_puzzles_not_piece_counts():
    before = [{"image_id": str(i), "exact": i < 2} for i in range(7)]
    after = [{"image_id": str(i), "exact": i >= 2} for i in range(7)]
    report = paired_comparison(before, after)
    assert len(report["recovered"]) == 5 and len(report["regressed"]) == 2
    assert report["two_sided_discordant_binomial_p"] == pytest.approx(0.453125)
    assert report["exact_rate_difference"] == pytest.approx(3 / 7)
    assert paired_comparison(before, after) == report
    assert paired_comparison(before, before)["two_sided_discordant_binomial_p"] == 1
    with pytest.raises(ValueError):
        paired_comparison(before, after[:-1])
    low, high = wilson_interval(45, 50)
    assert 0 <= low < 0.9 < high <= 1
    with pytest.raises(ValueError):
        aggregate([])


def test_image_only_pipeline_renders_a_permutation_and_preserves_weights():
    model = BoundaryCNN().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    pieces = [Image.new("RGB", (64, 64), (i * 10, 20, 30)) for i in range(25)]
    for scorer in (None, model):
        output = reconstruct(pieces, scorer, "beam")
        assert sorted(output["order_indices"]) == list(range(25))
        assert output["image"].size == (320, 320)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    with pytest.raises(ValueError, match="25 pieces"):
        score_pieces(pieces[:-1], model)
    with pytest.raises(ValueError, match="model.eval"):
        score_pieces(pieces, model.train())


def test_identical_pixels_can_be_correct_despite_different_piece_ids():
    truth = list(range(25))
    swapped = [1, 0] + list(range(2, 25))
    hashes = ["identical", "identical"] + [str(i) for i in range(2, 25)]
    assert swapped != truth
    assert appearance_metrics(swapped, truth, hashes) == {"pixel_position_correct": 25, "pixel_exact": True}
    hashes[1] = "different"
    assert appearance_metrics(swapped, truth, hashes) == {"pixel_position_correct": 23, "pixel_exact": False}


def test_complete_report_calibration_predictions_and_overlap_guards(tmp_path):
    root, fresh = tmp_path / "development", tmp_path / "fresh"
    index = []
    for number, split in enumerate(("train", "validation", "test", "test")):
        image_id = f"{number:016x}"
        source = tmp_path / f"image{number}.png"
        pixels = np.zeros((320, 320, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(320, dtype=np.uint16)[None, :] % 256
        pixels[:, :, 1] = np.arange(320, dtype=np.uint16)[:, None] % 256
        pixels[:, :, 2] = 20 * number
        Image.fromarray(pixels).save(source)
        relative = f"{split}/{image_id}/puzzle.json"
        target = root if number < 3 else fresh
        generate_puzzle(source, (target / relative).parent,
                        {"image_id": image_id, "split": split, "path": source.name, "sha256": sha256(source)}, 42)
        record = {"image_id": image_id, "split": split, "metadata_path": relative}
        if number < 3:
            index.append(record)
        else:
            (fresh / "index.json").write_text(json.dumps([record]))
    (root / "index.json").write_text(json.dumps(index))
    checkpoint = tmp_path / "model.pt"
    torch.save({"architecture": "boundary_cnn_v1", "model_state": BoundaryCNN().state_dict(),
                "config": {"direction_ids": DIRECTION_IDS, "train_ids": [index[0]["image_id"]],
                           "source_index_sha256": sha256(root / "index.json"),
                           "metadata_sha256": {index[1]["metadata_path"]: sha256(root / index[1]["metadata_path"])}}}, checkpoint)
    selection = tmp_path / "selection"
    selection.mkdir()
    (selection / "config.json").write_text(json.dumps({"method": SEARCH_METHOD, "checkpoint_sha256": sha256(checkpoint),
                                                       "development_ids": [r["image_id"] for r in index]}))
    (selection / "summary.json").write_text(json.dumps({"split": "validation", "selected_mode": "beam"}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"checkpoint_sha256": sha256(checkpoint), "selected_search": "beam",
                                    "validation_selection_sha256": sha256(selection / "summary.json"),
                                    "excluded_development_ids": [r["image_id"] for r in index],
                                    "candidate_ids": ["0000000000000003"], "count": 1}))
    output = tmp_path / "report"
    summary = run(root, fresh, checkpoint, selection, protocol, output)
    assert summary["cnn_weights_changed"] is False
    assert len(summary["cohorts"]["fresh_observed_holdout"]["methods"]) == 6
    assert (output / "report.md").exists() and (output / "comparison.png").exists()
    with np.load(output / "fresh_observed_holdout/pair_predictions.npz", allow_pickle=False) as pairs:
        assert pairs["labels"].shape == (1, 600) and pairs["labels"].sum() == 80
        labels = pairs["labels"].astype(bool)
        for scorer in ("classical", "cnn"):
            threshold = summary["calibration"][scorer]["threshold"]
            predicted = pairs[f"{scorer}_costs"] <= threshold
            expected = {"tp": int((predicted & labels).sum()), "fp": int((predicted & ~labels).sum()),
                        "tn": int((~predicted & ~labels).sum()), "fn": int((~predicted & labels).sum())}
            actual = summary["cohorts"]["fresh_observed_holdout"]["pairwise"][scorer]
            assert all(actual[k] == v for k, v in expected.items())
            assert summary["calibration"][scorer]["selection_split"] == "validation"
    for method, metrics in summary["cohorts"]["fresh_observed_holdout"]["methods"].items():
        rows = json.loads((output / "fresh_observed_holdout" / f"{method}.json").read_text())
        assert sorted(rows[0]["order_indices"]) == list(range(25))
        assert metrics["exact_puzzles"] == sum(r["exact"] for r in rows)
        assert metrics["fragment_position_accuracy"] == sum(r["position_correct"] for r in rows) / 25
    with pytest.raises(ValueError, match="empty"):
        run(root, fresh, checkpoint, selection, protocol, output)
    with pytest.raises(ValueError, match="overlap"):
        run(root, fresh, checkpoint, selection, protocol, root / "bad_output")
    (fresh / "index.json").write_text(json.dumps([index[2]]))
    with pytest.raises(ValueError, match="Holdout contains development"):
        run(root, fresh, checkpoint, selection, protocol, tmp_path / "bad_cohort")
