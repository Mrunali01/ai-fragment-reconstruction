"""Bounded CPU training with validation-only checkpoint selection."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import __version__ as pillow_version
import torch
from torch.utils.data import DataLoader, Subset

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.checkpoints import load_checkpoint, save_checkpoint
from jigsaw_ai.generate_puzzles import sha256
from jigsaw_ai.model_loss import pair_loss
from jigsaw_ai.pairs import PairDataset, read_index
from jigsaw_ai.siamese import SiameseCNN
from jigsaw_ai.tensor_data import collate_pairs


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 6
    batch_size: int = 64
    learning_rate: float = 0.001
    direction_weight: float = 1.0
    patience: int = 2
    threads: int = 2
    max_seconds: float = 600.0
    train_puzzles: int | None = None
    validation_puzzles: int | None = None

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "patience", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.direction_weight) or self.direction_weight < 0:
            raise ValueError("direction_weight must be finite and nonnegative")
        if not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be in 0..2**32-1")
        for limit in (self.train_puzzles, self.validation_puzzles):
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("Puzzle limits must be positive integers")


def pair_subset(root: Path, split: str, seed: int, limit: int | None) -> tuple[Subset, list[str]]:
    dataset = PairDataset(root, split, seed)
    ids = list(dict.fromkeys(record.puzzle_id for record in dataset.records))
    if limit is not None:
        if limit > len(ids):
            raise ValueError(f"Requested {limit} {split} puzzles, only {len(ids)} available")
        ids = ids[:limit]
    chosen = set(ids)
    return Subset(dataset, [i for i, record in enumerate(dataset.records) if record.puzzle_id in chosen]), ids


def run_epoch(model: SiameseCNN, loader: DataLoader, direction_weight: float,
              optimizer: torch.optim.Optimizer | None = None, deadline: float | None = None) -> dict:
    model.train(optimizer is not None)
    examples = positives = correct_binary = correct_direction = batches = 0
    binary_sum = direction_sum = 0.0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for batch in loader:
            if deadline is not None and batches and time.monotonic() >= deadline:
                break
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["first"], batch["second"])
            losses = pair_loss(outputs, batch["compatible"], batch["direction"], direction_weight)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("Nonfinite training/validation loss")
            if optimizer is not None:
                losses["loss"].backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError("Nonfinite or missing parameter gradients")
                optimizer.step()
            n = batch["compatible"].numel()
            mask = batch["compatible"] == 1
            p = int(mask.sum())
            examples += n
            positives += p
            binary_sum += losses["compatibility"].item() * n
            direction_sum += losses["direction"].item() * p
            correct_binary += int(((outputs["compatibility_logits"] >= 0) == mask).sum())
            if p:
                correct_direction += int((outputs["direction_logits"][mask].argmax(1) == batch["direction"][mask]).sum())
            batches += 1
            if optimizer is not None and batches % 100 == 0:
                print(f"Training progress: {batches} batches, {examples} pairs", flush=True)
    if not examples:
        raise ValueError("Epoch contains no examples")
    bce = binary_sum / examples
    ce = direction_sum / positives if positives else 0.0
    return {"examples": examples, "positive_examples": positives, "batches": batches,
            "compatibility_loss": bce, "direction_loss": ce, "loss": bce + direction_weight * ce,
            "compatibility_accuracy_at_0_5": correct_binary / examples,
            "positive_direction_accuracy": correct_direction / positives if positives else 0.0}


def plot_history(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "validation"):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], marker="o", label=split)
        axes[1].plot(epochs, [row[split]["positive_direction_accuracy"] for row in history], marker="o", label=split)
    axes[0].set(ylabel="BCE + weighted positive direction CE", title="Pair loss")
    axes[1].set(ylabel="Correct direction / positive pairs", ylim=(0, 1), title="Direction accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch (may end early at time budget)")
        axis.set_xticks(epochs)
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train(root: Path, output: Path, settings: TrainingConfig) -> dict:
    settings.validate()
    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError("Training output and puzzle directories must not overlap")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty; existing checkpoints are never overwritten by a new run")
    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    training, train_ids = pair_subset(root, "train", settings.seed, settings.train_puzzles)
    validation, val_ids = pair_subset(root, "validation", settings.seed, settings.validation_puzzles)
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation original images overlap")
    selected = set(train_ids + val_ids)
    config = {"training": asdict(settings), "device": "cpu", "architecture": "siamese_v1",
              "normalization": "RGB float32 / 255", "optimizer": "Adam",
              "selection_metric": "minimum validation BCE + weighted positive direction CE",
              "python_version": platform.python_version(), "torch_version": str(torch.__version__),
              "numpy_version": np.__version__, "pillow_version": pillow_version,
              "source_index_sha256": sha256(root / "index.json"), "train_ids": train_ids, "validation_ids": val_ids,
              "metadata_sha256": {r["metadata_path"]: sha256(root / r["metadata_path"])
                                   for r in read_index(root) if r["image_id"] in selected}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    generator = torch.Generator().manual_seed(settings.seed)
    train_loader = DataLoader(training, batch_size=settings.batch_size, shuffle=True, generator=generator,
                              num_workers=0, collate_fn=collate_pairs)
    val_loader = DataLoader(validation, batch_size=settings.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_pairs)
    model = SiameseCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    initial_validation = run_epoch(model, val_loader, settings.direction_weight)
    write_json(output / "initial_validation.json", initial_validation)
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    started = time.monotonic()
    deadline = started + settings.max_seconds
    stop_reason = "epoch_limit"
    for epoch in range(1, settings.epochs + 1):
        print(f"Starting epoch {epoch}/{settings.epochs}", flush=True)
        start = time.monotonic()
        train_stats = run_epoch(model, train_loader, settings.direction_weight, optimizer, deadline)
        val_stats = run_epoch(model, val_loader, settings.direction_weight)
        complete = train_stats["examples"] == len(training)
        history.append({"epoch": epoch, "train": train_stats, "validation": val_stats,
                        "epoch_complete": complete, "seconds": time.monotonic() - start})
        save_checkpoint(output / "last.pt", model, optimizer, epoch, config, val_stats)
        if val_stats["loss"] < best_loss:
            best_loss, best_epoch, stale = val_stats["loss"], epoch, 0
            save_checkpoint(output / "best.pt", model, optimizer, epoch, config, val_stats)
        else:
            stale += 1
        write_json(output / "history.json", history)
        print(f"Epoch {epoch}: train loss {train_stats['loss']:.4f}, validation loss {val_stats['loss']:.4f}, "
              f"validation direction {val_stats['positive_direction_accuracy']:.2%}; "
              f"{train_stats['examples']}/{len(training)} training examples", flush=True)
        if time.monotonic() >= deadline:
            stop_reason = "time_budget"
            break
        if stale >= settings.patience:
            stop_reason = "early_stopping"
            break
    best_model, checkpoint = load_checkpoint(output / "best.pt")
    reloaded = run_epoch(best_model, val_loader, settings.direction_weight)
    if reloaded != checkpoint["validation"]:
        raise RuntimeError("Reloaded checkpoint does not reproduce validation metrics")
    summary = {"epochs_recorded": len(history), "complete_epochs": sum(r["epoch_complete"] for r in history),
               "optimizer_steps": sum(r["train"]["batches"] for r in history), "best_epoch": best_epoch,
               "stop_reason": stop_reason, "training_seconds": time.monotonic() - started,
               "train_pairs": len(training), "validation_pairs": len(validation),
               "initial_validation": initial_validation, "best_validation": reloaded,
               "weights_changed": any(not torch.equal(initial[k], v) for k, v in model.state_dict().items()),
               "checkpoint_reload_verified": True}
    write_json(output / "summary.json", summary)
    plot_history(history, output / "learning_curves.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/training/siamese_cpu_v1"))
    for name, kind, default in (("epochs", int, 6), ("batch-size", int, 64), ("learning-rate", float, 0.001),
                                 ("direction-weight", float, 1.0), ("patience", int, 2), ("threads", int, 2),
                                 ("max-seconds", float, 600.0), ("seed", int, 42),
                                 ("train-puzzles", int, None), ("validation-puzzles", int, None)):
        parser.add_argument(f"--{name}", type=kind, default=default)
    args = parser.parse_args()
    try:
        settings = TrainingConfig(**{key: getattr(args, key) for key in TrainingConfig.__dataclass_fields__})
        print(json.dumps(train(args.source, args.output, settings), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
