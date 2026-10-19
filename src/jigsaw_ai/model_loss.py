"""Binary compatibility loss plus direction loss on positive pairs only."""

import math

import torch
from torch.nn import functional as F


def pair_loss(outputs: dict[str, torch.Tensor], compatible: torch.Tensor,
              direction: torch.Tensor, direction_weight: float = 1.0) -> dict[str, torch.Tensor]:
    if not math.isfinite(direction_weight) or direction_weight < 0:
        raise ValueError("Direction weight must be finite and nonnegative")
    binary, directional = outputs["compatibility_logits"], outputs["direction_logits"]
    if (binary.ndim != 1 or binary.numel() == 0 or directional.shape != (binary.shape[0], 4)
            or compatible.shape != binary.shape or direction.shape != binary.shape):
        raise ValueError("Expected logits [batch], [batch,4] and targets [batch]")
    if (not binary.is_floating_point() or directional.dtype != binary.dtype
            or compatible.dtype != binary.dtype or direction.dtype != torch.long):
        raise ValueError("Logits/compatibility targets require matching floating dtype; directions require int64")
    if any(t.device != binary.device for t in (directional, compatible, direction)):
        raise ValueError("All loss inputs must use the same device")
    if not all(torch.isfinite(t).all() for t in (binary, directional, compatible)):
        raise ValueError("Logits and compatibility targets must be finite")
    if not ((compatible == 0) | (compatible == 1)).all():
        raise ValueError("Compatibility targets must be 0 or 1")
    positive = compatible == 1
    if not (((direction >= 0) & (direction < 4)) | ~positive).all():
        raise ValueError("Positive pairs require direction IDs 0..3")
    if not ((direction == -1) | positive).all():
        raise ValueError("Negative pair directions must be -1")
    compatibility_loss = F.binary_cross_entropy_with_logits(binary, compatible)
    if positive.any():
        direction_loss = F.cross_entropy(directional[positive], direction[positive])
    else:
        # Keep a differentiable zero so negative-only batches backpropagate safely.
        direction_loss = directional[0, 0] * 0.0
    return {"loss": compatibility_loss + direction_weight * direction_loss,
            "compatibility": compatibility_loss, "direction": direction_loss}
