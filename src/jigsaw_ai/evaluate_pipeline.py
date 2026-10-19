"""Phase 8: reproducible image-to-reconstruction evaluation, with frozen methods."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.baseline_metrics import classification_metrics, pair_predictions, puzzle_metrics, select_threshold
from jigsaw_ai.baseline_reconstruction import arrangement_cost, render_arrangement
from jigsaw_ai.evaluation_report import aggregate, appearance_metrics, paired_comparison, save_report
from jigsaw_ai.evaluate_baseline import load_pieces
from jigsaw_ai.generate_puzzles import sha256, verify_puzzle
from jigsaw_ai.pairs import read_index
from jigsaw_ai.pipeline import assemble, score_pieces
from jigsaw_ai.search_experiments import METHOD as SEARCH_METHOD
from jigsaw_ai.solve_boundary import load_boundary_checkpoint

SCORERS = ("classical", "cnn")
SEARCHES = ("greedy", "swaps", "beam")
METHOD = {"version": 2, "scorers": list(SCORERS), "searches": list(SEARCHES),
          "primary": "cnn_beam", "pair_cost": "minimum of four directional costs",
          "threshold_selection": "validation maximum F1, lowest threshold on ties",
          "cnn_cost": "learned boundary distance divided by within-puzzle nonself mean",
          "search": SEARCH_METHOD, "training": "none", "test_tuning": False}


def cohort(root: Path, split: str) -> list[dict]:
    records = [r for r in read_index(root) if r["split"] == split]
    if not records:
        raise ValueError(f"No {split} puzzles")
    return records


def save_failure_preview(directory: Path, pieces: list[Image.Image], classical: list[int],
                         cnn: list[int], path: Path) -> None:
    canvas = Image.new("RGB", (1328, 350), "white")
    draw = ImageDraw.Draw(canvas)
    with Image.open(directory / "prepared.png") as target:
        panels = [("Ground truth (analysis only)", target),
                  ("Shuffled input", render_arrangement(pieces, list(range(25)))),
                  ("Classical greedy", render_arrangement(pieces, classical)),
                  ("CNN beam (frozen)", render_arrangement(pieces, cnn))]
        for index, (label, image) in enumerate(panels):
            x = 6 + 332 * index
            draw.text((x, 5), label, fill="black")
            canvas.paste(image, (x, 24))
    canvas.save(path)


def evaluate_cohort(root: Path, selected: list[dict], model, output: Path, name: str,
                    calibration: dict | None = None) -> tuple[dict, dict, list[dict], dict]:
    rows = {f"{scorer}_{search}": [] for scorer in SCORERS for search in SEARCHES}
    pairs = {scorer: [] for scorer in SCORERS}
    failures = []
    directory = output / name
    directory.mkdir()
    (directory / "predictions").mkdir()
    (directory / "failures").mkdir()
    for number, record in enumerate(selected, 1):
        source = (root / record["metadata_path"]).parent
        metadata = verify_puzzle(source)
        if metadata["image_id"] != record["image_id"] or metadata["split"] != record["split"]:
            raise ValueError("Puzzle metadata disagrees with index")
        pieces = load_pieces(source, metadata)
        results, costs_by_scorer = {}, {}
        for scorer in SCORERS:
            costs = score_pieces(pieces, model if scorer == "cnn" else None)
            costs_by_scorer[scorer] = costs
            for search in SEARCHES:
                order, cost = assemble(costs, search)
                results[f"{scorer}_{search}"] = {"image_id": record["image_id"],
                                                "order_indices": order, "mean_border_cost": cost}
        # Position labels enter only after all six image-only predictions exist.
        positions = [(p["original_row"], p["original_column"]) for p in metadata["pieces"]]
        true_order = sorted(range(25), key=lambda i: positions[i])
        fingerprints = [hashlib.sha256(piece.tobytes()).hexdigest() for piece in pieces]
        for method, prediction in results.items():
            scorer = method.split("_", 1)[0]
            prediction.update(puzzle_metrics(prediction["order_indices"], costs_by_scorer[scorer], positions))
            prediction.update(appearance_metrics(prediction["order_indices"], true_order, fingerprints))
            rows[method].append(prediction)
        for scorer in SCORERS:
            pairs[scorer].append(pair_predictions(costs_by_scorer[scorer], positions))
        primary = results["cnn_beam"]
        render_arrangement(pieces, primary["order_indices"]).save(directory / "predictions" / f"{record['image_id']}.png")
        if not primary["exact"]:
            true_cost = arrangement_cost(true_order, costs_by_scorer["cnn"])
            gap = true_cost - primary["mean_border_cost"]
            diagnosis = ("wrong arrangement has lower cost than ground truth" if gap > 1e-9 else
                         "ground truth has lower cost: search did not reach it" if gap < -1e-9 else
                         "ground truth and prediction have indistinguishable costs at tolerance 1e-9")
            hashes = {}
            for i, fingerprint in enumerate(fingerprints):
                hashes.setdefault(fingerprint, []).append(i)
            failures.append({"image_id": record["image_id"], "position_correct": primary["position_correct"],
                             "assembly_borders_correct": primary["assembly_borders_correct"],
                             "neighbor_correct": primary["neighbor_correct"], "prediction_cost": primary["mean_border_cost"],
                             "pixel_position_correct": primary["pixel_position_correct"], "pixel_exact": primary["pixel_exact"],
                             "ground_truth_cost": true_cost, "ground_truth_minus_prediction_cost": gap,
                             "objective_diagnosis": diagnosis,
                             "pixel_identical_piece_groups": [ids for ids in hashes.values() if len(ids) > 1]})
            save_failure_preview(source, pieces, results["classical_greedy"]["order_indices"],
                                 primary["order_indices"], directory / "failures" / f"{record['image_id']}.png")
        if number % 15 == 0 or number == len(selected):
            print(f"End-to-end {name}: {number}/{len(selected)} puzzles", flush=True)
    if calibration is None:
        calibration = {scorer: {"threshold": select_threshold([pair for puzzle in pairs[scorer] for pair in puzzle]),
                                "selection_split": "validation", "examples": 600 * len(selected)} for scorer in SCORERS}
    report = {"methods": {method: aggregate(predictions) for method, predictions in rows.items()},
              "pairwise": {scorer: classification_metrics([pair for puzzle in pairs[scorer] for pair in puzzle],
                                                           calibration[scorer]["threshold"]) for scorer in SCORERS}}
    np.savez_compressed(directory / "pair_predictions.npz", image_ids=np.array([r["image_id"] for r in selected]),
                        labels=np.array([[label for _, label in puzzle] for puzzle in pairs["classical"]], dtype=np.uint8),
                        classical_costs=np.array([[cost for cost, _ in puzzle] for puzzle in pairs["classical"]]),
                        cnn_costs=np.array([[cost for cost, _ in puzzle] for puzzle in pairs["cnn"]]))
    for method, predictions in rows.items():
        write_json(directory / f"{method}.json", predictions)
    write_json(directory / "summary.json", report)
    write_json(directory / "failures.json", failures)
    return report, calibration, failures, rows


def run(development: Path, holdout: Path, checkpoint: Path, selection: Path,
        protocol_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Phase 8 output must be new/empty")
    for source in (development, holdout, selection):
        if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
            raise ValueError("Report output must not overlap inputs")
    if checkpoint.resolve().is_relative_to(output.resolve()) or protocol_path.resolve().is_relative_to(output.resolve()):
        raise ValueError("Report output must not contain checkpoint or protocol")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, payload = load_boundary_checkpoint(checkpoint)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoint_hash = sha256(checkpoint)
    configuration = json.loads((selection / "config.json").read_text(encoding="utf-8"))
    frozen = json.loads((selection / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (configuration["method"] != SEARCH_METHOD or frozen["split"] != "validation" or frozen["selected_mode"] != "beam"
            or configuration["checkpoint_sha256"] != checkpoint_hash or protocol["checkpoint_sha256"] != checkpoint_hash
            or protocol["selected_search"] != "beam" or protocol["validation_selection_sha256"] != sha256(selection / "summary.json")):
        raise ValueError("Frozen selection, protocol, and checkpoint disagree")
    if sha256(development / "index.json") != payload["config"]["source_index_sha256"]:
        raise ValueError("Development index differs from training")
    for path, expected in payload["config"]["metadata_sha256"].items():
        if sha256(development / path) != expected:
            raise ValueError("Training/validation metadata changed")
    validation, fresh = cohort(development, "validation"), cohort(holdout, "test")
    known = {r["image_id"] for r in read_index(development)}
    if known != set(configuration["development_ids"]) or known != set(protocol["excluded_development_ids"]):
        raise ValueError("Development exclusions differ from frozen selection")
    fresh_ids = {r["image_id"] for r in fresh}
    if (fresh_ids & known or len(fresh) != protocol["count"]
            or not fresh_ids <= set(protocol["candidate_ids"])):
        raise ValueError("Holdout contains development images or violates acquisition protocol")
    if {r["image_id"] for r in validation} & set(payload["config"]["train_ids"]):
        raise ValueError("Validation images overlap training")
    inputs = {"method": METHOD, "checkpoint_sha256": checkpoint_hash,
              "development_index_sha256": sha256(development / "index.json"),
              "holdout_index_sha256": sha256(holdout / "index.json"),
              "selection_sha256": sha256(selection / "summary.json"), "protocol_sha256": sha256(protocol_path),
              "torch_version": str(torch.__version__), "threads": 2,
              "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                ("pipeline.py", "evaluate_pipeline.py", "evaluation_report.py", "boundary_model.py",
                                 "group_search.py", "baseline_metrics.py", "baseline_reconstruction.py", "edge_baseline.py")},
              "validation_metadata_sha256": {r["metadata_path"]: sha256(development / r["metadata_path"]) for r in validation},
              "holdout_metadata_sha256": {r["metadata_path"]: sha256(holdout / r["metadata_path"]) for r in fresh}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", inputs)
    validation_report, calibration, _, _ = evaluate_cohort(development, validation, model, output, "validation")
    write_json(output / "calibration.json", {"method": METHOD, "checkpoint_sha256": checkpoint_hash,
               "source_index_sha256": sha256(development / "index.json"), "thresholds": calibration})
    fresh_report, _, failures, rows = evaluate_cohort(holdout, fresh, model, output, "fresh_observed_holdout", calibration)
    comparisons = {f"cnn_beam_vs_{baseline}": paired_comparison(rows[baseline], rows["cnn_beam"])
                   for baseline in ("cnn_swaps", "classical_greedy", "classical_beam")}
    cohorts = {"validation": validation_report, "fresh_observed_holdout": fresh_report}
    summary = {"cohorts": cohorts, "paired_comparisons": comparisons, "cnn_weights_changed": False,
               "primary_method": "cnn_beam", "fresh_cohort_is_now_observed": True,
               "failed_primary_fresh_puzzles": len(failures), "calibration": calibration}
    if (sha256(checkpoint) != checkpoint_hash
            or not all(torch.equal(initial_weights[k], v) for k, v in model.state_dict().items())):
        raise RuntimeError("Checkpoint or in-memory model weights changed during evaluation")
    write_json(output / "summary.json", summary)
    save_report(output, cohorts, calibration, comparisons, failures)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--holdout", type=Path, default=Path("data/fresh_puzzles_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cnn_experiments/boundary_v1/ranking/best.pt"))
    parser.add_argument("--selection", type=Path, default=Path("artifacts/search_experiments/validation_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("data/fresh_holdout_v1/protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase8/final"))
    args = parser.parse_args()
    try:
        run(args.development, args.holdout, args.checkpoint, args.selection, args.protocol, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Phase 8 evaluation failed: {error}\n")
