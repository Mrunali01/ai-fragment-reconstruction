"""Evaluate a frozen checkpoint; fit pair threshold on validation only."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.baseline_metrics import classification_metrics, select_threshold
from jigsaw_ai.checkpoints import load_checkpoint
from jigsaw_ai.evaluate_baseline import load_pieces
from jigsaw_ai.generate_puzzles import sha256, verify_puzzle
from jigsaw_ai.model_inference import neighbor_queries, score_puzzle
from jigsaw_ai.pairs import DIRECTION_IDS, read_index

METHOD = {"architecture": "siamese_v1", "pair_score": "sigmoid compatibility",
          "neighbor_score": "P(compatible) times P(direction given compatible)",
          "direction_ids": DIRECTION_IDS, "normalization": "RGB float32 / 255"}


def save_failure_examples(root: Path, errors: list[dict], path: Path) -> None:
    canvas = Image.new("RGB", (720, 32 + len(errors) * 90), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), "First piece       Predicted        Ground truth      Second relative to first; scores are not guarantees", fill="black")
    for row, error in enumerate(errors):
        y = 30 + row * 90
        directory = root / error["split"] / error["image_id"] / "pieces"
        for column, key in enumerate(("first_id", "prediction_id", "target_id")):
            with Image.open(directory / f"{error[key]}.png") as image:
                canvas.paste(image, (8 + column * 100, y))
        draw.text((310, y + 5), f"{error['direction']} | {error['image_id']}", fill="black")
        draw.text((310, y + 23), f"predicted joint score: {error['predicted_joint_score']:.4f}", fill="black")
        draw.text((310, y + 41), f"true neighbor joint score: {error['target_joint_score']:.4f}", fill="black")
    canvas.save(path)


def evaluate_model(root: Path, checkpoint_path: Path, output: Path, split: str,
                   calibration_path: Path | None = None, baseline_root: Path | None = None) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("Evaluate validation or test only")
    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError("Evaluation output and source directories must not overlap")
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(checkpoint_path)
    index_hash, checkpoint_hash = sha256(root / "index.json"), sha256(checkpoint_path)
    training = checkpoint["config"]
    if training["source_index_sha256"] != index_hash:
        raise ValueError("Checkpoint was trained with a different puzzle index")
    for path, expected in training["metadata_sha256"].items():
        if sha256(root / path) != expected:
            raise ValueError("Training/validation source metadata changed after training")
    threshold = None
    if split == "test":
        if calibration_path is None:
            raise ValueError("Test evaluation requires validation calibration")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (calibration["split"] != "validation" or calibration["checkpoint_sha256"] != checkpoint_hash
                or calibration["source_index_sha256"] != index_hash or calibration["method"] != METHOD):
            raise ValueError("Calibration must match this checkpoint, dataset, and validation method")
        threshold = calibration["cost_threshold"]
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("Calibration threshold must be within 0..1")
    selected = [r for r in read_index(root) if r["split"] == split]
    if not selected:
        raise ValueError(f"No puzzles found for {split}")
    if set(r["image_id"] for r in selected) & set(training["train_ids"]):
        raise ValueError("Evaluation images overlap training images")
    metadata_hashes = {r["metadata_path"]: sha256(root / r["metadata_path"]) for r in selected}
    config = {"method": METHOD, "split": split, "checkpoint_sha256": checkpoint_hash,
              "source_index_sha256": index_hash, "metadata_sha256": metadata_hashes,
              "calibration_sha256": sha256(calibration_path) if split == "test" else None}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Evaluation output uses different inputs; choose a new directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Evaluation output must be empty")
        output.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
    records, predictions, errors = [], [], []
    direction_correct = positive_pairs = 0
    for number, item in enumerate(selected, 1):
        directory = (root / item["metadata_path"]).parent
        metadata = verify_puzzle(directory)
        if metadata["image_id"] != item["image_id"] or metadata["split"] != split:
            raise ValueError("Puzzle metadata disagrees with index")
        scores = score_puzzle(model, load_pieces(directory, metadata))
        if not all(torch.isfinite(t).all() for t in scores.values()):
            raise RuntimeError("Nonfinite inference scores")
        positions = [(p["original_row"], p["original_column"]) for p in metadata["pieces"]]
        for i, (row, column) in enumerate(positions):
            for j, (other_row, other_column) in enumerate(positions):
                if i == j:
                    continue
                dr, dc = other_row - row, other_column - column
                label = int(abs(dr) + abs(dc) == 1)
                predictions.append((1 - float(scores["compatibility"][i, j]), label))
                if label:
                    direction = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}[dr, dc]
                    direction_correct += int(int(scores["conditional_direction"][i, j].argmax()) == DIRECTION_IDS[direction])
                    positive_pairs += 1
        queries = neighbor_queries(scores, positions)
        correct = sum(q["prediction_index"] == q["target_index"] for q in queries)
        total = len(queries)
        for query in queries:
            if query["prediction_index"] != query["target_index"] and len(errors) < 8:
                error = {"image_id": item["image_id"], "split": split,
                         "direction": query["direction"], "predicted_joint_score": query["predicted_joint_score"],
                         "target_joint_score": query["target_joint_score"]}
                for name, key in (("first_id", "first_index"), ("prediction_id", "prediction_index"), ("target_id", "target_index")):
                    error[name] = metadata["pieces"][query[key]]["piece_id"]
                errors.append(error)
        records.append({"image_id": item["image_id"], "neighbor_correct": correct, "neighbor_queries": total,
                        "neighbor_prediction_accuracy": correct / total})
        if number % 15 == 0 or number == len(selected):
            print(f"Scored {number}/{len(selected)} {split} puzzles", flush=True)
    if split == "validation":
        threshold = select_threshold(predictions)
        write_json(output / "calibration.json", {"split": split, "method": METHOD,
                   "source_index_sha256": index_hash, "checkpoint_sha256": checkpoint_hash,
                   "cost_threshold": threshold, "probability_threshold": 1 - threshold,
                   "selection": "maximum validation pair F1 over all ordered non-self pairs"})
    summary = {"split": split, "puzzles": len(selected), "pair_examples": len(predictions),
               "checkpoint_epoch": checkpoint["epoch"], "compatibility_probability_threshold": 1 - threshold,
               "pairwise": classification_metrics(predictions, threshold),
               "positive_direction_accuracy": direction_correct / positive_pairs,
               "neighbor_correct": sum(r["neighbor_correct"] for r in records),
               "neighbor_queries": sum(r["neighbor_queries"] for r in records),
               "neighbor_prediction_accuracy": sum(r["neighbor_correct"] for r in records) / sum(r["neighbor_queries"] for r in records)}
    if baseline_root is not None:
        baseline_config = json.loads((baseline_root / split / "config.json").read_text(encoding="utf-8"))
        if baseline_config["source_index_sha256"] != index_hash or baseline_config["metadata_sha256"] != metadata_hashes:
            raise ValueError("Baseline comparison requires the same held-out puzzles and metadata")
        baseline = json.loads((baseline_root / split / "summary.json").read_text(encoding="utf-8"))
        summary["classical_baseline"] = {"pairwise": baseline["pairwise"],
                                          "neighbor_prediction_accuracy": baseline["neighbor_prediction_accuracy"]}
    write_json(output / "summary.json", summary)
    write_json(output / "per_puzzle.json", records)
    write_json(output / "failure_analysis.json", {
        "lowest_neighbor_accuracy": sorted(records, key=lambda r: (r["neighbor_prediction_accuracy"], r["image_id"]))[:5],
        "first_neighbor_errors": errors,
        "note": "Neural assembly is not implemented in Phase 6; no position/exact reconstruction metrics are reported."})
    if errors:
        save_failure_examples(root, errors, output / "failure_examples.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training/siamese_cpu_v1/best.pt"))
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path, default=Path("artifacts/model_evaluation/validation/calibration.json"))
    parser.add_argument("--baseline-root", type=Path, default=Path("data/baseline"))
    args = parser.parse_args()
    try:
        output = args.output or Path("artifacts/model_evaluation") / args.split
        print(json.dumps(evaluate_model(args.source, args.checkpoint, output, args.split,
                                        args.calibration, args.baseline_root), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Model evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
