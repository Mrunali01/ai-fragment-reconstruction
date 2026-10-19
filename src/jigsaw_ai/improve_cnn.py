"""Validation-only, reproducible boundary CNN experiments on the existing splits."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.baseline_metrics import puzzle_metrics
from jigsaw_ai.baseline_reconstruction import greedy_reconstruct, render_arrangement
from jigsaw_ai.boundary_model import BoundaryCNN, OPPOSITE, boundary_strips, ranking_targets, training_loss
from jigsaw_ai.evaluate_baseline import load_pieces
from jigsaw_ai.generate_puzzles import sha256, verify_puzzle
from jigsaw_ai.pairs import DIRECTION_IDS, read_index
from jigsaw_ai.tensor_data import image_tensor


def load_split(root: Path, split: str) -> list[dict]:
    records = []
    for item in read_index(root):
        if item["split"] != split:
            continue
        directory = (root / item["metadata_path"]).parent
        metadata = verify_puzzle(directory)
        if metadata["image_id"] != item["image_id"] or metadata["split"] != split:
            raise ValueError("Puzzle metadata and index disagree")
        positions = [(p["original_row"], p["original_column"]) for p in metadata["pieces"]]
        pieces = torch.stack([image_tensor(p) for p in load_pieces(directory, metadata)])
        records.append({"image_id": item["image_id"], "strips": boundary_strips(pieces),
                        "targets": ranking_targets(positions), "positions": positions,
                        "metadata_path": item["metadata_path"], "directory": directory})
        if len(records) % 50 == 0:
            print(f"Cached {len(records)} {split} puzzles", flush=True)
    if not records:
        raise ValueError(f"No {split} puzzles")
    return records


def raw_costs(strips: torch.Tensor) -> torch.Tensor:
    """The classical touching-edge RGB MAE, expressed on the 0..1 pixel scale."""
    return torch.stack([(strips[:, d, :, :, 0][:, None] - strips[:, opposite, :, :, 0][None]).abs().mean((-1, -2))
                        for d, opposite in enumerate(OPPOSITE)])


def cost_dict(costs: torch.Tensor) -> dict:
    result = costs.clone()
    result[:, range(25), range(25)] = float("inf")
    return {direction: result[index].tolist() for direction, index in DIRECTION_IDS.items()}


def refine_swaps(order: list[int], costs: dict, max_rounds: int = 20) -> tuple[list[int], float]:
    """Best improving two-piece swap, accepting only strict objective decreases."""
    if len(order) != 25 or set(order) != set(range(25)) or max_rounds < 0:
        raise ValueError("Expected a complete arrangement and nonnegative round limit")
    right, down = np.array(costs["RIGHT"]), np.array(costs["DOWN"])
    left_positions = np.array([i for i in range(25) if i % 5 < 4])
    upper_positions = np.arange(20)
    def objective(orders):
        return (right[orders[:, left_positions], orders[:, left_positions + 1]].sum(1)
                + down[orders[:, upper_positions], orders[:, upper_positions + 5]].sum(1)) / 40
    current = np.array(order)
    score = float(objective(current[None])[0])
    swaps = [(i, j) for i in range(25) for j in range(i + 1, 25)]
    for _ in range(max_rounds):
        candidates = np.tile(current, (len(swaps), 1))
        for row, (i, j) in enumerate(swaps):
            candidates[row, i], candidates[row, j] = current[j], current[i]
        scores = objective(candidates)
        best = int(scores.argmin())
        if scores[best] >= score - 1e-12:
            break
        current, score = candidates[best], float(scores[best])
    return current.tolist(), score


@torch.inference_mode()
def validate(model: BoundaryCNN, records: list[dict]) -> dict:
    model.eval()
    correct, losses = 0, []
    for record in records:
        logits = model(record["strips"])
        masked = logits.masked_fill(torch.eye(25, dtype=torch.bool).unsqueeze(0), -float("inf"))
        valid = record["targets"] >= 0
        correct += int((masked.argmax(-1)[valid] == record["targets"][valid]).sum())
        losses.append(float(torch.nn.functional.cross_entropy(masked[valid], record["targets"][valid])))
    return {"neighbor_correct": correct, "neighbor_queries": 80 * len(records),
            "neighbor_prediction_accuracy": correct / (80 * len(records)), "ranking_loss": sum(losses) / len(losses)}


def fit(train: list[dict], validation: list[dict], output: Path, objective: str, epochs: int, provenance: dict) -> dict:
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    model = BoundaryCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    initial = validate(model, validation)
    best, best_epoch = initial, 0
    history = []
    def save(epoch, metrics):
        temporary = output / "best.pt.tmp"
        torch.save({"architecture": "boundary_cnn_v1", "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "epoch": epoch, "metrics": metrics,
                    "config": {**provenance, "objective": objective, "epochs": epochs,
                               "seed": 42, "learning_rate": 0.001, "direction_ids": DIRECTION_IDS},
                    "torch_rng_state": torch.get_rng_state(), "sampling_rng_state": generator.get_state()}, temporary)
        temporary.replace(output / "best.pt")
    output.mkdir()
    save(0, initial)
    write_json(output / "initial_validation.json", initial)
    print(f"{objective}: initial validation neighbor accuracy {initial['neighbor_prediction_accuracy']:.4%}", flush=True)
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        indices = torch.randperm(len(train), generator=generator).tolist()
        for step, index in enumerate(indices, 1):
            record = train[index]
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model(record["strips"]), record["targets"], objective, generator)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite boundary training loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite boundary gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += float(loss.detach())
            if step % 100 == 0:
                print(f"{objective} epoch {epoch}/{epochs}, puzzle {step}/{len(train)}", flush=True)
        metrics = validate(model, validation)
        row = {"epoch": epoch, "training_loss": total_loss / len(train), "validation": metrics}
        history.append(row)
        key = lambda m: (m["neighbor_correct"], -m["ranking_loss"])
        if key(metrics) > key(best):
            best, best_epoch = metrics, epoch
            save(epoch, metrics)
        write_json(output / "history.json", history)
        print(f"{objective} epoch {epoch}: validation neighbor accuracy {metrics['neighbor_prediction_accuracy']:.4%}", flush=True)
    payload = torch.load(output / "best.pt", weights_only=True)
    model.load_state_dict(payload["model_state"])
    reloaded = validate(model, validation)
    if reloaded != best:
        raise RuntimeError("Checkpoint validation changed on reload")
    changed = any(not torch.equal(initial_state[k], v) for k, v in model.state_dict().items())
    summary = {"objective": objective, "epochs": epochs, "optimizer_steps": epochs * len(train),
               "initial_validation": initial, "best_validation": best, "best_epoch": best_epoch,
               "checkpoint_reload_verified": True, "selected_weights_changed": changed,
               "training_seconds": time.monotonic() - started}
    write_json(output / "summary.json", summary)
    return summary


@torch.inference_mode()
def assembly_experiments(model: BoundaryCNN, records: list[dict], output: Path,
                        cnn_weights: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0),
                        refinements: tuple[bool, ...] = (False, True)) -> list[dict]:
    model.eval()
    cached = []
    off_diagonal = ~torch.eye(25, dtype=torch.bool)
    for record in records:
        logits = model(record["strips"])
        neural = (model.bias - logits) / model.log_scale.exp()
        classical = raw_costs(record["strips"])
        cached.append((neural / neural[:, off_diagonal].mean().clamp_min(1e-8),
                       classical / classical[:, off_diagonal].mean().clamp_min(1e-8)))
    reports = []
    # alpha=1 is CNN alone; alpha=0 is classical. Intermediate values are hybrids.
    for alpha in cnn_weights:
        for refine in refinements:
            rows = []
            for record, (neural, classical) in zip(records, cached):
                costs = cost_dict(alpha * neural + (1 - alpha) * classical)
                order, cost = greedy_reconstruct(costs)
                if refine:
                    order, cost = refine_swaps(order, costs)
                metrics = puzzle_metrics(order, costs, record["positions"])
                rows.append({"image_id": record["image_id"], "order_indices": order,
                             "mean_border_cost": cost, **metrics})
            n = len(rows)
            report = {"cnn_weight": alpha, "refinement": refine, "puzzles": n,
                      "neighbor_prediction_accuracy": sum(r["neighbor_correct"] for r in rows) / (80 * n),
                      "fragment_position_accuracy": sum(r["position_correct"] for r in rows) / (25 * n),
                      "assembly_neighbor_accuracy": sum(r["assembly_borders_correct"] for r in rows) / (40 * n),
                      "exact_puzzles": sum(r["exact"] for r in rows)}
            name = f"cnn_{alpha}_swaps_{refine}"
            write_json(output / f"{name}.json", {"summary": report, "results": rows})
            reports.append(report)
            print(f"Assembly {name}: positions {report['fragment_position_accuracy']:.2%}, exact {report['exact_puzzles']}/{n}", flush=True)
    write_json(output / "assembly_comparison.json", reports)
    return reports


def checkpoint_ablation(root: Path, output: Path) -> list[dict]:
    """Separate initialization benefits from learning in full puzzle assembly."""
    torch.set_num_threads(2)
    provenance = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if provenance["source_index_sha256"] != sha256(root / "index.json"):
        raise ValueError("Experiment index changed")
    for path, expected in provenance["metadata_sha256"].items():
        if sha256(root / path) != expected:
            raise ValueError("Experiment metadata changed")
    records = load_split(root, "validation")
    reports = []
    for name in ("initial", "random", "hard", "ranking"):
        torch.manual_seed(42)
        model = BoundaryCNN()
        if name != "initial":
            payload = torch.load(output / name / "best.pt", weights_only=True)
            model.load_state_dict(payload["model_state"], strict=True)
        directory = output / "checkpoint_ablation" / name
        directory.mkdir(parents=True, exist_ok=True)
        for report in assembly_experiments(model, records, directory, (1.0,), (False, True)):
            reports.append({"checkpoint": name, **report})
    write_json(output / "checkpoint_assembly_comparison.json", reports)
    return reports


def run(root: Path, output: Path, random_epochs: int = 4, hard_epochs: int = 4, ranking_epochs: int = 12) -> dict:
    if min(random_epochs, hard_epochs, ranking_epochs) < 1:
        raise ValueError("All experiment epoch counts must be positive")
    if (output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve())):
        raise ValueError("Experiment output and source must not overlap")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment output must be new/empty")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    train, validation = load_split(root, "train"), load_split(root, "validation")
    if {r["image_id"] for r in train} & {r["image_id"] for r in validation}:
        raise ValueError("Training and validation images overlap")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"source_index_sha256": sha256(root / "index.json"),
                  "metadata_sha256": {r["metadata_path"]: sha256(root / r["metadata_path"]) for r in train + validation},
                  "train_ids": [r["image_id"] for r in train], "validation_ids": [r["image_id"] for r in validation],
                  "test_used": False, "threads": 2, "torch_version": str(torch.__version__),
                  "selection": "validation neighbor accuracy, then ranking loss; initial model eligible"}
    write_json(output / "config.json", provenance)
    reports = []
    for objective, epochs in (("random", random_epochs), ("hard", hard_epochs), ("ranking", ranking_epochs)):
        reports.append(fit(train, validation, output / objective, objective, epochs, provenance))
        write_json(output / "training_comparison.json", reports)
    selected = max(reports, key=lambda r: (r["best_validation"]["neighbor_correct"], -r["best_validation"]["ranking_loss"]))
    checkpoint_path = output / selected["objective"] / "best.pt"
    payload = torch.load(checkpoint_path, weights_only=True)
    model = BoundaryCNN()
    model.load_state_dict(payload["model_state"])
    assembly = assembly_experiments(model, validation, output)
    best_assembly = max(assembly, key=lambda r: (r["exact_puzzles"], r["fragment_position_accuracy"], r["assembly_neighbor_accuracy"]))
    summary = {"split": "validation", "test_used": False, "training": reports,
               "selected_cnn": selected["objective"], "checkpoint_sha256": sha256(checkpoint_path),
               "assembly": assembly, "selected_assembly": best_assembly,
               "note": "Validation-selected results; fresh unseen evaluation is required before claiming final generalization."}
    write_json(output / "summary.json", summary)
    # Save actual predictions for the best pure CNN arrangement and overall selection.
    pure = max((r for r in assembly if r["cnn_weight"] == 1),
               key=lambda r: (r["exact_puzzles"], r["fragment_position_accuracy"]))
    for label, configuration in (("pure_cnn", pure), ("selected", best_assembly)):
        name = f"cnn_{configuration['cnn_weight']}_swaps_{configuration['refinement']}"
        rows = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))["results"]
        directory = output / label
        directory.mkdir()
        for record, row in zip(validation, rows):
            metadata = json.loads((record["directory"] / "puzzle.json").read_text(encoding="utf-8"))
            pieces = load_pieces(record["directory"], metadata)
            render_arrangement(pieces, row["order_indices"]).save(directory / f"{record['image_id']}.png")
    checkpoint_ablation(root, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cnn_experiments/boundary_v1"))
    parser.add_argument("--random-epochs", type=int, default=4)
    parser.add_argument("--hard-epochs", type=int, default=4)
    parser.add_argument("--ranking-epochs", type=int, default=12)
    args = parser.parse_args()
    try:
        summary = run(args.source, args.output, args.random_epochs, args.hard_epochs, args.ranking_epochs)
        print(json.dumps({k: summary[k] for k in ("selected_cnn", "selected_assembly", "test_used")}, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"CNN experiments failed: {error}\n")


if __name__ == "__main__":
    main()
