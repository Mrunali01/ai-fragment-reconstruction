"""Evaluate the frozen classical baseline on validation or test puzzles."""

import argparse
import json
from pathlib import Path
import platform

from PIL import Image, ImageDraw, __version__ as pillow_version

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.baseline_metrics import classification_metrics, pair_predictions, puzzle_metrics, select_threshold
from jigsaw_ai.baseline_reconstruction import greedy_reconstruct, render_arrangement
from jigsaw_ai.edge_baseline import compatibility_costs
from jigsaw_ai.generate_puzzles import sha256, verify_puzzle
from jigsaw_ai.pairs import read_index

METHOD = {"version": 1, "metric": "touching-edge RGB mean absolute difference",
          "solver": "row-major greedy; all 25 top-left anchors; lowest mean 40-border cost",
          "pair_cost": "minimum of four directional costs", "tie_break": "lowest shuffled piece index"}


def load_pieces(directory: Path, metadata: dict) -> list[Image.Image]:
    pieces = []
    for entry in metadata["pieces"]:
        with Image.open(directory / entry["path"]) as image:
            pieces.append(image.copy())
    return pieces


def save_preview(directory: Path, pieces: list[Image.Image], order: list[int], path: Path, title: str) -> None:
    canvas = Image.new("RGB", (1000, 375), "white")
    draw = ImageDraw.Draw(canvas)
    with Image.open(directory / "prepared.png") as target:
        panels = [("Ground truth (evaluation only)", target),
                  ("Shuffled input", render_arrangement(pieces, list(range(25)))),
                  ("Edge-baseline prediction", render_arrangement(pieces, order))]
        for index, (label, image) in enumerate(panels):
            x = 8 + index * 332
            draw.text((x, 8), label, fill="black")
            canvas.paste(image, (x, 30))
    draw.text((8, 355), title, fill="black")
    canvas.save(path)


def evaluate(root: Path, output: Path, split: str, calibration_path: Path | None = None) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("Evaluate validation or test only")
    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError("Evaluation output and puzzle directories must not overlap")
    index_hash = sha256(root / "index.json")
    threshold = None
    if split == "test":
        if calibration_path is None:
            raise ValueError("Test evaluation requires validation calibration")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (calibration["split"] != "validation" or calibration["method"] != METHOD
                or calibration["source_index_sha256"] != index_hash):
            raise ValueError("Calibration must come from this dataset's validation split and the same method")
        threshold = calibration["threshold"]
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 255:
            raise ValueError("Calibration threshold must be within 0..255")
    selected = [row for row in read_index(root) if row["split"] == split]
    if not selected:
        raise ValueError(f"No puzzles found for {split}")
    config = {"method": METHOD, "split": split, "source_index_sha256": index_hash,
              "python_version": platform.python_version(), "pillow_version": pillow_version,
              "calibration_sha256": sha256(calibration_path) if split == "test" else None,
              "metadata_sha256": {row["metadata_path"]: sha256(root / row["metadata_path"]) for row in selected}}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing evaluation has different inputs; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Evaluation output directory is not empty")
        output.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
    (output / "predictions").mkdir(exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    results, predictions = [], []
    for number, item in enumerate(selected):
        directory = (root / item["metadata_path"]).parent
        metadata = verify_puzzle(directory)
        if metadata["image_id"] != item["image_id"] or metadata["split"] != split:
            raise ValueError("Puzzle metadata and index disagree")
        pieces = load_pieces(directory, metadata)
        costs = compatibility_costs(pieces)
        # Only pixel-derived costs enter the solver; labels are used below for evaluation.
        order, edge_cost = greedy_reconstruct(costs)
        positions = [(p["original_row"], p["original_column"]) for p in metadata["pieces"]]
        metrics = puzzle_metrics(order, costs, positions)
        predictions.extend(pair_predictions(costs, positions))
        results.append({"image_id": item["image_id"], "split": split,
                        "order": [metadata["pieces"][i]["piece_id"] for i in order],
                        "mean_edge_cost": edge_cost, **metrics})
        render_arrangement(pieces, order).save(output / "predictions" / f"{item['image_id']}.png")
        if number < 3:
            title = (f"{split}/{item['image_id']} | Position: {metrics['position_accuracy']:.1%}"
                     f" | Neighbor ranking: {metrics['neighbor_prediction_accuracy']:.1%}")
            save_preview(directory, pieces, order, output / "previews" / f"{number + 1:02}.png", title)
        if (number + 1) % 15 == 0 or number + 1 == len(selected):
            print(f"Evaluated {number + 1}/{len(selected)} {split} puzzles", flush=True)
    if split == "validation":
        threshold = select_threshold(predictions)
        write_json(output / "calibration.json", {"split": split, "threshold": threshold,
                   "selection": "maximum validation pair F1; lowest threshold on ties",
                   "source_index_sha256": index_hash, "method": METHOD})
    n = len(results)
    exact_count = sum(r["exact"] for r in results)
    summary = {"split": split, "puzzles": n, "pair_examples": len(predictions),
               "threshold": threshold, "pairwise": classification_metrics(predictions, threshold),
               "neighbor_prediction_accuracy": sum(r["neighbor_correct"] for r in results) / (80 * n),
               "fragment_position_accuracy": sum(r["position_correct"] for r in results) / (25 * n),
               "assembly_neighbor_accuracy": sum(r["assembly_borders_correct"] for r in results) / (40 * n),
               "exact_puzzles": exact_count, "exact_reconstruction_rate": exact_count / n,
               "mean_edge_cost": sum(r["mean_edge_cost"] for r in results) / n}
    write_json(output / "results.json", results)
    write_json(output / "summary.json", summary)
    write_json(output / "failure_analysis.json", {
        "failed_puzzles": [r["image_id"] for r in results if not r["exact"]],
        "lowest_position_accuracy": [r["image_id"] for r in sorted(results, key=lambda r: (r["position_accuracy"], r["image_id"]))[:5]],
        "limitations": ["Greedy early mistakes cannot be undone", "Uniform colors and repeated patterns give ambiguous edges",
                        "Low edge cost is not a calibrated correctness probability"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path, default=Path("data/baseline/validation/calibration.json"))
    args = parser.parse_args()
    try:
        output = args.output or Path("data/baseline") / args.split
        print(json.dumps(evaluate(args.source, output, args.split, args.calibration), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Baseline evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
