"""Plot actual validation histories and compare short versus longer ranking runs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from jigsaw_ai.acquire_images import write_json


def plot(output: Path) -> dict:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    histories = {}
    for objective in ("random", "hard", "ranking"):
        history = json.loads((output / objective / "history.json").read_text(encoding="utf-8"))
        initial = json.loads((output / objective / "initial_validation.json").read_text(encoding="utf-8"))
        rows = [{"epoch": 0, "validation": initial}] + history
        histories[objective] = rows
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [100 * r["validation"]["neighbor_prediction_accuracy"] for r in rows], marker=".", label=objective)
        axes[1].plot(epochs, [r["validation"]["ranking_loss"] for r in rows], marker=".", label=objective)
    axes[0].set_ylabel("Validation neighbor accuracy (%)")
    axes[1].set_ylabel("Validation ranking cross entropy")
    for axis in axes:
        axis.set_xlabel("Complete epochs (0 = explicit initial edge prior)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output / "learning_curves.png", dpi=160)
    plt.close(figure)
    key = lambda row: (row["validation"]["neighbor_correct"], -row["validation"]["ranking_loss"])
    ranking = histories["ranking"]
    short = max((r for r in ranking if r["epoch"] <= 4), key=key)
    full = max(ranking, key=key)
    comparison = {"ranking_best_through_epoch_4": short, "ranking_best_full_run": full,
                  "additional_neighbor_correct": full["validation"]["neighbor_correct"] - short["validation"]["neighbor_correct"],
                  "test_used": False}
    write_json(output / "longer_training_comparison.json", comparison)
    return comparison


def gallery(output: Path, source: Path, old_predictions: Path) -> Path:
    """Use the first three validation inputs, without selecting by outcome."""
    selected = json.loads((output / "config.json").read_text(encoding="utf-8"))["validation_ids"][:3]
    canvas = Image.new("RGB", (1320, 355 * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, image_id in enumerate(selected):
        panels = (("Ground truth", source / "validation" / image_id / "prepared.png"),
                  ("Original whole-piece CNN", old_predictions / f"{image_id}.png"),
                  ("New pure boundary CNN", output / "pure_cnn" / f"{image_id}.png"),
                  ("Validation-selected method", output / "selected" / f"{image_id}.png"))
        for column, (label, path) in enumerate(panels):
            x, y = column * 330 + 5, row * 355
            draw.text((x, y + 2), label, fill="black")
            with Image.open(path) as image:
                canvas.paste(image, (x, y + 22))
        draw.text((5, row * 355 + 343), image_id, fill="black")
    path = output / "reconstruction_comparison.png"
    canvas.save(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cnn_experiments/boundary_v1"))
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--old-predictions", type=Path, default=Path("artifacts/reconstruction/validation/predictions"))
    parser.add_argument("--gallery", action="store_true")
    args = parser.parse_args()
    print(json.dumps(plot(args.output), indent=2))
    if args.gallery:
        print(gallery(args.output, args.source, args.old_predictions))
