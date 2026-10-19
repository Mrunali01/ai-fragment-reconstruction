"""Evaluate frozen neural assembly on validation, then held-out test puzzles."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.baseline_metrics import puzzle_metrics
from jigsaw_ai.baseline_reconstruction import greedy_reconstruct, render_arrangement
from jigsaw_ai.checkpoints import load_checkpoint
from jigsaw_ai.evaluate_baseline import load_pieces
from jigsaw_ai.generate_puzzles import sha256, verify_puzzle
from jigsaw_ai.model_inference import score_puzzle
from jigsaw_ai.neural_reconstruction import directional_costs
from jigsaw_ai.pairs import read_index

METHOD = {"version": 1, "cost": "1 - P(compatible) * P(direction given compatible)",
          "solver": "row-major greedy; all 25 anchors; lowest mean 40-border cost",
          "tie_break": "lowest shuffled piece index, then lexicographic arrangement",
          "border_directions": "RIGHT from left piece; DOWN from upper piece",
          "rotations": False, "grid": [5, 5]}
METRICS = ("neighbor_prediction_accuracy", "fragment_position_accuracy",
           "assembly_neighbor_accuracy", "exact_puzzles", "exact_reconstruction_rate")


def save_preview(directory: Path, pieces: list[Image.Image], order: list[int], path: Path) -> None:
    canvas = Image.new("RGB", (1000, 350), "white")
    draw = ImageDraw.Draw(canvas)
    with Image.open(directory / "prepared.png") as target:
        panels = [("Ground truth (evaluation only)", target),
                  ("Shuffled input", render_arrangement(pieces, list(range(25)))),
                  ("CNN greedy prediction", render_arrangement(pieces, order))]
        for index, (label, image) in enumerate(panels):
            x = 8 + index * 332
            draw.text((x, 6), label, fill="black")
            canvas.paste(image, (x, 25))
    canvas.save(path)


def evaluate(root: Path, checkpoint_path: Path, output: Path, split: str,
             validation_path: Path | None = None, baseline_root: Path | None = None) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("Evaluate validation or test only")
    if (output.resolve().is_relative_to(root.resolve())
            or root.resolve().is_relative_to(output.resolve())
            or checkpoint_path.resolve().is_relative_to(output.resolve())):
        raise ValueError("Output must not overlap source data or contain the checkpoint")
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(checkpoint_path)
    index_hash, checkpoint_hash = sha256(root / "index.json"), sha256(checkpoint_path)
    training = checkpoint["config"]
    if training["source_index_sha256"] != index_hash:
        raise ValueError("Checkpoint uses a different puzzle index")
    for path, expected in training["metadata_sha256"].items():
        if sha256(root / path) != expected:
            raise ValueError("Training/validation metadata changed after training")
    selected = [item for item in read_index(root) if item["split"] == split]
    if not selected:
        raise ValueError(f"No puzzles found for {split}")
    if {item["image_id"] for item in selected} & set(training["train_ids"]):
        raise ValueError("Evaluation images overlap training images")
    if split == "test":
        if validation_path is None:
            raise ValueError("Test requires a completed validation reconstruction run")
        validation = json.loads((validation_path / "config.json").read_text(encoding="utf-8"))
        completed = json.loads((validation_path / "summary.json").read_text(encoding="utf-8"))
        if (validation["split"] != "validation" or completed["split"] != "validation"
                or validation["method"] != METHOD or validation["checkpoint_sha256"] != checkpoint_hash
                or validation["source_index_sha256"] != index_hash):
            raise ValueError("Validation must match this frozen method, checkpoint, and dataset")
        for path, expected in validation["metadata_sha256"].items():
            if sha256(root / path) != expected:
                raise ValueError("Validation metadata changed after reconstruction")
    metadata_hashes = {item["metadata_path"]: sha256(root / item["metadata_path"]) for item in selected}
    baseline = None
    if baseline_root is not None:
        config = json.loads((baseline_root / split / "config.json").read_text(encoding="utf-8"))
        if config["source_index_sha256"] != index_hash or config["metadata_sha256"] != metadata_hashes:
            raise ValueError("Baseline comparison requires identical puzzles and metadata")
        baseline = json.loads((baseline_root / split / "summary.json").read_text(encoding="utf-8"))
    config = {"method": METHOD, "split": split, "source_index_sha256": index_hash,
              "checkpoint_sha256": checkpoint_hash, "metadata_sha256": metadata_hashes,
              "validation_config_sha256": sha256(validation_path / "config.json") if split == "test" else None,
              "baseline_summary_sha256": sha256(baseline_root / split / "summary.json") if baseline_root else None}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing output uses different inputs; choose a new directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output must be empty")
        output.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
    (output / "predictions").mkdir(exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    results = []
    for number, item in enumerate(selected, 1):
        directory = (root / item["metadata_path"]).parent
        metadata = verify_puzzle(directory)
        if metadata["image_id"] != item["image_id"] or metadata["split"] != split:
            raise ValueError("Puzzle metadata disagrees with index")
        pieces = load_pieces(directory, metadata)
        costs = directional_costs(score_puzzle(model, pieces))
        # The solver sees scores only. Position labels enter metrics after assembly.
        order, cost = greedy_reconstruct(costs)
        positions = [(p["original_row"], p["original_column"]) for p in metadata["pieces"]]
        metrics = puzzle_metrics(order, costs, positions)
        results.append({"image_id": item["image_id"], "split": split,
                        "order": [metadata["pieces"][i]["piece_id"] for i in order],
                        "mean_border_cost": cost, **metrics})
        render_arrangement(pieces, order).save(output / "predictions" / f"{item['image_id']}.png")
        if number <= 3:
            save_preview(directory, pieces, order, output / "previews" / f"{number:02}.png")
        if number % 15 == 0 or number == len(selected):
            print(f"Reconstructed {number}/{len(selected)} {split} puzzles", flush=True)
    n = len(results)
    exact = sum(r["exact"] for r in results)
    summary = {"split": split, "puzzles": n, "checkpoint_epoch": checkpoint["epoch"],
               "neighbor_prediction_accuracy": sum(r["neighbor_correct"] for r in results) / (80 * n),
               "fragment_position_accuracy": sum(r["position_correct"] for r in results) / (25 * n),
               "assembly_neighbor_accuracy": sum(r["assembly_borders_correct"] for r in results) / (40 * n),
               "exact_puzzles": exact, "exact_reconstruction_rate": exact / n,
               "mean_border_cost": sum(r["mean_border_cost"] for r in results) / n}
    if baseline is not None:
        summary["classical_baseline"] = {key: baseline[key] for key in METRICS}
    write_json(output / "results.json", results)
    write_json(output / "summary.json", summary)
    write_json(output / "failure_analysis.json", {
        "failed_puzzles": [r["image_id"] for r in results if not r["exact"]],
        "lowest_position_accuracy": sorted(results, key=lambda r: (r["position_accuracy"], r["image_id"]))[:5],
        "limitations": ["Greedy placements cannot be undone", "Scores need not agree in opposite directions",
                        "Mean border score is not a probability of correct reconstruction"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training/siamese_cpu_v1/best.pt"))
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation", type=Path, default=Path("artifacts/reconstruction/validation"))
    parser.add_argument("--baseline-root", type=Path, default=Path("data/baseline"))
    args = parser.parse_args()
    try:
        output = args.output or Path("artifacts/reconstruction") / args.split
        print(json.dumps(evaluate(args.source, args.checkpoint, output, args.split,
                                  args.validation, args.baseline_root), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Reconstruction failed: {error}\n")


if __name__ == "__main__":
    main()
