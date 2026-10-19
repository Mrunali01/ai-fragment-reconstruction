"""Small ordered-pair Siamese CNN with shared weights and two logit heads."""

import torch
from torch import nn


class SiameseCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
        )
        self.pair_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.compatibility_head = nn.Linear(128, 1)
        self.direction_head = nn.Linear(128, 4)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> dict[str, torch.Tensor]:
        if (first.ndim != 4 or first.shape != second.shape or first.shape[0] == 0
                or tuple(first.shape[1:]) != (3, 64, 64)):
            raise ValueError("Inputs must have matching nonempty [batch, 3, 64, 64] shapes")
        parameter = next(self.parameters())
        for image in (first, second):
            if image.dtype != parameter.dtype or image.device != parameter.device:
                raise ValueError("Inputs must match the model's floating dtype and device")
            if not torch.isfinite(image).all():
                raise ValueError("Input tensors must contain only finite values")
        # The SAME module encodes both branches. Concatenation retains input order.
        first_features = self.encoder(first)
        second_features = self.encoder(second)
        features = self.pair_head(torch.cat((first_features, second_features), dim=1))
        return {
            "compatibility_logits": self.compatibility_head(features).squeeze(-1),
            "direction_logits": self.direction_head(features),
        }
