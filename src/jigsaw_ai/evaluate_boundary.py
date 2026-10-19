"""Audit validation-frozen boundary CNN settings on the previously seen test set."""

import argparse
import json
from pathlib import Path

import torch

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.baseline_reconstruction import render_arrangement
from jigsaw_ai.evaluate_baseline import load_pieces
from jigsaw_ai.generate_puzzles import sha256
from jigsaw_ai.improve_cnn import assembly_experiments, load_split
from jigsaw_ai.solve_boundary import load_boundary_checkpoint


def evaluate(root: Path, experiments: Path, output: Path) -> dict:
    if (output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve())
            or output.resolve().is_relative_to(experiments.resolve()) or experiments.resolve().is_relative_to(output.resolve())):
        raise ValueError("Audit output must not overlap data or experiment directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Audit output must be new/empty")
    summary = json.loads((experiments / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((experiments / "config.json").read_text(encoding="utf-8"))
    if summary["split"] != "validation" or summary["test_used"]:
        raise ValueError("Selection must come exclusively from validation")
    if config["source_index_sha256"] != sha256(root / "index.json"):
        raise ValueError("Experiment index changed")
    for path, expected in config["metadata_sha256"].items():
        if sha256(root / path) != expected:
            raise ValueError("Training/validation metadata changed")
    if summary["selected_cnn"] not in ("random", "hard", "ranking"):
        raise ValueError("Unknown selected model")
    checkpoint_path = experiments / summary["selected_cnn"] / "best.pt"
    if sha256(checkpoint_path) != summary["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    torch.set_num_threads(2)
    model, _ = load_boundary_checkpoint(checkpoint_path)
    # All settings are chosen before reading or scoring test images.
    pure = max((r for r in summary["assembly"] if r["cnn_weight"] == 1),
               key=lambda r: (r["exact_puzzles"], r["fragment_position_accuracy"]))
    settings = {"pure_cnn": pure, "selected": summary["selected_assembly"]}
    records = load_split(root, "test")
    if {r["image_id"] for r in records} & set(config["train_ids"] + config["validation_ids"]):
        raise ValueError("Test images overlap development images")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {
        "split": "test", "selection_summary_sha256": sha256(experiments / "summary.json"),
        "checkpoint_sha256": sha256(checkpoint_path), "settings": settings,
        "metadata_sha256": {r["metadata_path"]: sha256(root / r["metadata_path"]) for r in records},
        "note": "Previously inspected test set: confirmation only, not a fresh independent final test."})
    reports = {}
    for label, setting in settings.items():
        directory = output / label
        directory.mkdir()
        reports[label] = assembly_experiments(model, records, directory,
                                              (setting["cnn_weight"],), (setting["refinement"],))[0]
        name = f"cnn_{setting['cnn_weight']}_swaps_{setting['refinement']}"
        rows = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))["results"]
        (directory / "predictions").mkdir()
        for record, row in zip(records, rows):
            metadata = json.loads((record["directory"] / "puzzle.json").read_text(encoding="utf-8"))
            render_arrangement(load_pieces(record["directory"], metadata), row["order_indices"]).save(
                directory / "predictions" / f"{record['image_id']}.png")
    result = {"split": "test", "selection_split": "validation", "results": reports,
              "note": "No model or setting changes were made using these test results."}
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--experiments", type=Path, default=Path("artifacts/cnn_experiments/boundary_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cnn_experiments/test_audit_v1"))
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.source, args.experiments, args.output), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Frozen boundary audit failed: {error}\n")
