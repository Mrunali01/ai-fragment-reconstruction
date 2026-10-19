"""Check an UNTRAINED CNN on real training pairs; no optimizer or training loop."""

import argparse
import json
from pathlib import Path

import torch

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.model_loss import pair_loss
from jigsaw_ai.pairs import PairDataset
from jigsaw_ai.siamese import SiameseCNN
from jigsaw_ai.tensor_data import collate_pairs


def inspect_model(root: Path, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    model = SiameseCNN()
    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    dataset = PairDataset(root, "train", seed)
    positive = [i for i, r in enumerate(dataset.records) if r.compatible][:4]
    negative = [i for i, r in enumerate(dataset.records) if not r.compatible][:4]
    indices = positive + negative
    batch = collate_pairs([dataset[i] for i in indices])
    batch["first"].requires_grad_(True)
    batch["second"].requires_grad_(True)
    outputs = model(batch["first"], batch["second"])
    losses = pair_loss(outputs, batch["compatible"], batch["direction"])
    losses["loss"].backward()
    gradients_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    input_gradients = all(image.grad is not None and bool(torch.isfinite(image.grad).all())
                          and bool(image.grad.abs().sum() > 0) for image in (batch["first"], batch["second"]))
    unchanged = all(torch.equal(initial[name], value) for name, value in model.state_dict().items())
    negative_batch = collate_pairs([dataset[i] for i in negative])
    with torch.no_grad():
        negative_outputs = model(negative_batch["first"], negative_batch["second"])
        negative_losses = pair_loss(negative_outputs, negative_batch["compatible"], negative_batch["direction"])
        single_outputs = model(batch["first"][:1], batch["second"][:1])
    if not gradients_finite or not input_gradients or not unchanged or negative_losses["direction"].item() != 0:
        raise RuntimeError("Model computation checks failed")
    return {
        "status": "untrained; computation verification only", "seed": seed,
        "torch_version": torch.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": len(indices), "positive_pairs": len(positive), "negative_pairs": len(negative),
        "input_shape": list(batch["first"].shape),
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "single_output_shapes": {key: list(value.shape) for key, value in single_outputs.items()},
        "initial_losses": {key: value.item() for key, value in losses.items()},
        "negative_only_direction_loss": negative_losses["direction"].item(),
        "all_parameter_gradients_finite": gradients_finite, "both_input_gradients_nonzero_and_finite": input_gradients,
        "weights_unchanged": unchanged, "optimizer_steps": 0,
        "puzzle_ids": sorted({dataset.records[i].puzzle_id for i in indices}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase5_model_check.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        torch.set_num_threads(2)
        report = inspect_model(args.source, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Model check failed: {error}\n")


if __name__ == "__main__":
    main()
