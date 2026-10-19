"""Versioned model checkpoints with explicit architecture and label metadata."""

from pathlib import Path

import torch

from jigsaw_ai.acquire_images import replace_file
from jigsaw_ai.pairs import DIRECTION_IDS
from jigsaw_ai.siamese import SiameseCNN


def save_checkpoint(path: Path, model: SiameseCNN, optimizer: torch.optim.Optimizer,
                    epoch: int, config: dict, metrics: dict) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"schema_version": 1, "architecture": "siamese_v1", "direction_ids": DIRECTION_IDS,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "config": config, "validation": metrics,
                "torch_rng_state": torch.get_rng_state()}, temporary)
    replace_file(temporary, path)


def load_checkpoint(path: Path) -> tuple[SiameseCNN, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != 1 or checkpoint.get("architecture") != "siamese_v1"
            or checkpoint.get("direction_ids") != DIRECTION_IDS):
        raise ValueError("Unsupported checkpoint architecture, schema, or direction mapping")
    model = SiameseCNN()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
