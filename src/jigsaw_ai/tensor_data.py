"""Convert existing Pillow pair samples into model tensors, without augmentation."""

from PIL import Image
import numpy as np
import torch


def image_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB" or image.size != (64, 64):
        raise ValueError("Expected an RGB 64x64 fragment")
    pixels = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).div(255).contiguous()


def collate_pairs(samples: list[dict]) -> dict[str, torch.Tensor]:
    """A DataLoader collate function; -1 marks directions excluded from loss."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    compatibility, directions = [], []
    for sample in samples:
        label, direction = sample["compatible"], sample["direction"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError("Compatibility labels must be integer 0 or 1")
        if label == 1:
            if type(direction) is not int or direction not in range(4):
                raise ValueError("Positive pairs require a direction ID from 0 to 3")
        elif direction is not None:
            raise ValueError("Negative pairs must have direction None")
        compatibility.append(label)
        directions.append(direction if label else -1)
    return {
        "first": torch.stack([image_tensor(s["first"]) for s in samples]),
        "second": torch.stack([image_tensor(s["second"]) for s in samples]),
        "compatible": torch.tensor(compatibility, dtype=torch.float32),
        "direction": torch.tensor(directions, dtype=torch.long),
    }
