"""Reconstruct 25 upright RGB square fragments with a saved boundary CNN."""

import argparse
from pathlib import Path

from PIL import Image
import torch

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.baseline_reconstruction import greedy_reconstruct, render_arrangement
from jigsaw_ai.boundary_model import BoundaryCNN, boundary_strips
from jigsaw_ai.generate_puzzles import sha256
from jigsaw_ai.group_search import group_refine
from jigsaw_ai.improve_cnn import cost_dict, raw_costs, refine_swaps
from jigsaw_ai.pairs import DIRECTION_IDS
from jigsaw_ai.tensor_data import image_tensor


def load_boundary_checkpoint(path: Path) -> tuple[BoundaryCNN, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("architecture") != "boundary_cnn_v1"
            or checkpoint.get("config", {}).get("direction_ids") != DIRECTION_IDS):
        raise ValueError("Expected a boundary_cnn_v1 checkpoint with the current direction mapping")
    model = BoundaryCNN()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite parameters")
    return model.eval(), checkpoint


@torch.inference_mode()
def solve(pieces_path: Path, checkpoint_path: Path, output: Path,
          cnn_weight: float = 1.0, refine: bool = False, search: str | None = None) -> dict:
    if not 0 <= cnn_weight <= 1:
        raise ValueError("CNN weight must be within 0..1")
    mode = search or ("swaps" if refine else "greedy")
    if mode not in ("greedy", "swaps", "groups", "beam"):
        raise ValueError("Unknown search mode")
    if (output.resolve().is_relative_to(pieces_path.resolve())
            or pieces_path.resolve().is_relative_to(output.resolve())
            or checkpoint_path.resolve().is_relative_to(output.resolve())):
        raise ValueError("Output must not overlap input pieces or contain the checkpoint")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new/empty")
    paths = sorted(pieces_path.glob("*.png"))
    if len(paths) != 25:
        raise ValueError("Expected exactly 25 PNG pieces")
    pieces = []
    for path in paths:
        with Image.open(path) as image:
            # Validation is explicit: no silent resizing or rotation of fragments.
            image_tensor(image)
            pieces.append(image.copy())
    torch.set_num_threads(2)
    model, _ = load_boundary_checkpoint(checkpoint_path)
    strips = boundary_strips(torch.stack([image_tensor(p) for p in pieces]))
    logits = model(strips)
    neural = (model.bias - logits) / model.log_scale.exp()
    classical = raw_costs(strips)
    mask = ~torch.eye(25, dtype=torch.bool)
    neural /= neural[:, mask].mean().clamp_min(1e-8)
    classical /= classical[:, mask].mean().clamp_min(1e-8)
    combined = cnn_weight * neural + (1 - cnn_weight) * classical
    if not torch.isfinite(combined).all():
        raise ValueError("Nonfinite reconstruction costs")
    costs = cost_dict(combined)
    order, cost = greedy_reconstruct(costs)
    if mode != "greedy":
        order, cost = refine_swaps(order, costs)
    if mode in ("groups", "beam"):
        order, cost = group_refine(order, costs, mode)
    output.mkdir(parents=True, exist_ok=True)
    render_arrangement(pieces, order).save(output / "reconstruction.png")
    result = {"checkpoint_sha256": sha256(checkpoint_path), "cnn_weight": cnn_weight,
              "refinement": mode != "greedy", "search": mode, "grid": [5, 5], "mean_border_cost": cost,
              "input_sha256": {p.name: sha256(p) for p in paths},
              "order": [paths[i].name for i in order],
              "note": "No ground truth supplied: reconstruction accuracy cannot be calculated."}
    write_json(output / "arrangement.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cnn-weight", type=float, default=1.0)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--search", choices=("greedy", "swaps", "groups", "beam"))
    args = parser.parse_args()
    try:
        result = solve(args.pieces, args.checkpoint, args.output, args.cnn_weight, args.refine, args.search)
        print(f"Saved reconstruction using {len(result['order'])} unique pieces to {args.output}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Boundary reconstruction failed: {error}\n")


if __name__ == "__main__":
    main()
