"""Compare two fixed search extensions on validation; audit a fresh frozen cohort."""

import argparse
import json
from pathlib import Path

import torch

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.baseline_metrics import puzzle_metrics
from jigsaw_ai.baseline_reconstruction import greedy_reconstruct, render_arrangement
from jigsaw_ai.evaluate_baseline import load_pieces
from jigsaw_ai.generate_puzzles import sha256
from jigsaw_ai.group_search import group_refine
from jigsaw_ai.improve_cnn import cost_dict, load_split, refine_swaps
from jigsaw_ai.pairs import read_index
from jigsaw_ai.solve_boundary import load_boundary_checkpoint

METHOD = {"version": 1, "scorer": "frozen boundary_cnn_v1, CNN only",
          "moves": "single swaps, line cycles, whole-line swaps, global shifts, disjoint 2x2 block swaps",
          "groups_rounds": 20, "beam_width": 4, "beam_depth": 4,
          "selection": "validation exact count, then positions, then borders; incumbent wins ties"}


@torch.inference_mode()
def evaluate(root: Path, checkpoint: Path, output: Path, frozen_validation: Path | None = None,
             holdout_protocol: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Search output must be new/empty")
    if (output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve())
            or checkpoint.resolve().is_relative_to(output.resolve())):
        raise ValueError("Output must not overlap inputs or contain checkpoint")
    torch.set_num_threads(2)
    model, payload = load_boundary_checkpoint(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    if frozen_validation is None:
        split = "validation"
        if sha256(root / "index.json") != payload["config"]["source_index_sha256"]:
            raise ValueError("Validation source differs from training index")
        for path, expected in payload["config"]["metadata_sha256"].items():
            if sha256(root / path) != expected:
                raise ValueError("Training/validation metadata changed")
        modes = ("swaps", "groups", "beam")
        excluded_ids = [r["image_id"] for r in read_index(root)]
    else:
        split = "test"
        frozen = json.loads((frozen_validation / "summary.json").read_text(encoding="utf-8"))
        configuration = json.loads((frozen_validation / "config.json").read_text(encoding="utf-8"))
        if (frozen["split"] != "validation" or configuration["method"] != METHOD
                or configuration["checkpoint_sha256"] != checkpoint_hash):
            raise ValueError("Frozen validation does not match method and checkpoint")
        modes = tuple(dict.fromkeys(("swaps", frozen["selected_mode"])))
        excluded_ids = configuration["development_ids"]
    records = load_split(root, split)
    forbidden = set(payload["config"]["train_ids"])
    if split == "test":
        forbidden.update(excluded_ids)
    if forbidden & {r["image_id"] for r in records}:
        raise ValueError("Evaluation cohort overlaps development images")
    if holdout_protocol is not None:
        protocol = json.loads(holdout_protocol.read_text(encoding="utf-8"))
        if (frozen_validation is None or protocol["validation_selection_sha256"] != sha256(frozen_validation / "summary.json")
                or protocol["checkpoint_sha256"] != checkpoint_hash
                or protocol["selected_search"] != frozen["selected_mode"] or protocol["count"] != len(records)):
            raise ValueError("Holdout protocol does not match the frozen selection and cohort size")
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"split": split, "method": METHOD, "checkpoint_sha256": checkpoint_hash,
                     "source_index_sha256": sha256(root / "index.json"),
                     "metadata_sha256": {r["metadata_path"]: sha256(root / r["metadata_path"]) for r in records},
                     "development_ids": excluded_ids, "modes": list(modes),
                     "holdout_protocol_sha256": sha256(holdout_protocol) if holdout_protocol else None,
                     "frozen_selection_sha256": sha256(frozen_validation / "summary.json") if frozen_validation else None}
    write_json(output / "config.json", configuration)
    rows = {mode: [] for mode in modes}
    for number, record in enumerate(records, 1):
        logits = model(record["strips"])
        distance = (model.bias - logits) / model.log_scale.exp()
        distance /= distance[:, ~torch.eye(25, dtype=torch.bool)].mean().clamp_min(1e-8)
        if not torch.isfinite(distance).all():
            raise ValueError("Nonfinite scores")
        costs = cost_dict(distance)
        order, _ = greedy_reconstruct(costs)
        incumbent, incumbent_cost = refine_swaps(order, costs)
        for mode in modes:
            selected, cost = (incumbent, incumbent_cost) if mode == "swaps" else group_refine(incumbent, costs, mode)
            if cost > incumbent_cost + 1e-10:
                raise RuntimeError("Search increased the cost")
            metrics = puzzle_metrics(selected, costs, record["positions"])
            rows[mode].append({"image_id": record["image_id"], "order_indices": selected,
                               "mean_border_cost": cost, **metrics})
        if number % 15 == 0 or number == len(records):
            print(f"Search audit {number}/{len(records)} {split} puzzles", flush=True)
    reports = []
    for mode in modes:
        n = len(records)
        reports.append({"mode": mode, "puzzles": n, "exact_puzzles": sum(r["exact"] for r in rows[mode]),
                        "position_correct": sum(r["position_correct"] for r in rows[mode]),
                        "fragment_position_accuracy": sum(r["position_correct"] for r in rows[mode]) / (25 * n),
                        "assembly_borders_correct": sum(r["assembly_borders_correct"] for r in rows[mode]),
                        "assembly_neighbor_accuracy": sum(r["assembly_borders_correct"] for r in rows[mode]) / (40 * n),
                        "neighbor_prediction_accuracy": sum(r["neighbor_correct"] for r in rows[mode]) / (80 * n)})
        write_json(output / f"{mode}.json", rows[mode])
    if split == "validation":
        selected_mode = max(reports, key=lambda r: (r["exact_puzzles"], r["position_correct"], r["assembly_borders_correct"]))["mode"]
    else:
        selected_mode = frozen["selected_mode"]
    selected_rows = rows[selected_mode]
    incumbent_rows = rows["swaps"]
    transitions = {"recovered": [a["image_id"] for a, b in zip(incumbent_rows, selected_rows) if not a["exact"] and b["exact"]],
                   "regressed": [a["image_id"] for a, b in zip(incumbent_rows, selected_rows) if a["exact"] and not b["exact"]]}
    summary = {"split": split, "selected_mode": selected_mode, "reports": reports, "transitions": transitions,
               "cnn_retrained": False, "checkpoint_sha256": checkpoint_hash,
               "note": "Test selection is frozen from validation; fresh results do not change settings."}
    write_json(output / "summary.json", summary)
    directory = output / "predictions"
    directory.mkdir()
    for record, row in zip(records, selected_rows):
        metadata = json.loads((record["directory"] / "puzzle.json").read_text(encoding="utf-8"))
        render_arrangement(load_pieces(record["directory"], metadata), row["order_indices"]).save(directory / f"{record['image_id']}.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cnn_experiments/boundary_v1/ranking/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/search_experiments/validation_v1"))
    parser.add_argument("--frozen-validation", type=Path)
    parser.add_argument("--holdout-protocol", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.source, args.checkpoint, args.output, args.frozen_validation, args.holdout_protocol), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Search experiment failed: {error}\n")
