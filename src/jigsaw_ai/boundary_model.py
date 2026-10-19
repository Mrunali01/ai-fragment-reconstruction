"""A shared, trainable edge encoder and nonnegative learned boundary distance."""

import torch
from torch import nn
from torch.nn import functional as F

from jigsaw_ai.pairs import DIRECTION_IDS

OPPOSITE = [DIRECTION_IDS[d] for d in ("RIGHT", "LEFT", "DOWN", "UP")]


def boundary_strips(pieces: torch.Tensor) -> torch.Tensor:
    """Return [pieces, LEFT/RIGHT/UP/DOWN, RGB, along-edge, inward-depth]."""
    if pieces.ndim != 4 or tuple(pieces.shape[1:]) != (3, 64, 64):
        raise ValueError("Expected RGB pieces with shape [N,3,64,64]")
    return torch.stack((pieces[..., :4], pieces[..., -4:].flip(-1),
                        pieces[:, :, :4, :].transpose(-1, -2),
                        pieces[:, :, -4:, :].transpose(-1, -2).flip(-1)), dim=1)


class BoundaryCNN(nn.Module):
    """Learn edge features and how much each feature/depth should contribute.

    Identity initialization is an explicit RGB-continuity prior, not a trained
    result. A saved initial validation score makes that prior measurable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, (3, 1), padding=(1, 0)), nn.ReLU(),
                                     nn.Conv2d(8, 8, 1), nn.ReLU())
        with torch.no_grad():
            first, second = self.encoder[0], self.encoder[2]
            first.weight.normal_(0, 0.02)
            first.bias.zero_()
            first.weight[:3].zero_()
            for c in range(3):
                first.weight[c, c, 1, 0] = 1
            second.weight.zero_()
            second.bias.zero_()
            for c in range(8):
                second.weight[c, c, 0, 0] = 1
        weights = torch.tensor([1.0, 0.03, 0.01, 0.003]).repeat(8, 1)
        weights[3:] *= 0.03
        self.raw_weights = nn.Parameter(torch.log(torch.expm1(weights)))
        self.log_scale = nn.Parameter(torch.tensor(3.0))
        self.bias = nn.Parameter(torch.tensor(3.0))

    def forward(self, strips: torch.Tensor) -> torch.Tensor:
        """Return direction-specific logits [4,25,25]; larger means better."""
        if strips.shape != (25, 4, 3, 64, 4) or not torch.isfinite(strips).all():
            raise ValueError("Expected finite strips with shape [25,4,3,64,4]")
        encoded = self.encoder(strips.reshape(100, 3, 64, 4)).reshape(25, 4, 8, 64, 4)
        weights = F.softplus(self.raw_weights)
        weights = weights / weights.sum()
        logits = []
        for direction, opposite in enumerate(OPPOSITE):
            a, b = encoded[:, direction], encoded[:, opposite]
            differences = (a[:, None] - b[None, :]).abs().mean(dim=-2)
            distance = (differences * weights).sum(dim=(-1, -2))
            logits.append(self.bias - self.log_scale.exp() * distance)
        return torch.stack(logits)


def ranking_targets(positions: list[tuple[int, int]]) -> torch.Tensor:
    if len(positions) != 25 or set(positions) != {(r, c) for r in range(5) for c in range(5)}:
        raise ValueError("Expected complete 5x5 positions")
    lookup = {p: i for i, p in enumerate(positions)}
    return torch.tensor([[lookup.get((r + dr, c + dc), -1) for r, c in positions]
                         for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0))])


def training_loss(logits: torch.Tensor, targets: torch.Tensor, objective: str,
                  generator: torch.Generator) -> torch.Tensor:
    """Rank all 24 candidates, or balance positives with sampled negatives."""
    if objective not in ("random", "hard", "ranking"):
        raise ValueError("Unknown boundary training objective")
    rows = targets >= 0
    masked = logits.masked_fill(torch.eye(25, dtype=torch.bool).unsqueeze(0), -float("inf"))
    candidates, labels = masked[rows], targets[rows]
    if objective == "ranking":
        return F.cross_entropy(candidates, labels)
    positive = candidates.gather(1, labels[:, None]).squeeze(1)
    negatives = candidates.clone()
    negatives.scatter_(1, labels[:, None], -float("inf"))
    if objective == "hard":
        choices = negatives.topk(8, dim=1).indices
        selected = torch.randint(8, (len(labels),), generator=generator)
        indices = choices[torch.arange(len(labels)), selected]
    else:
        random = torch.rand(negatives.shape, generator=generator)
        random[~torch.isfinite(negatives)] = -1
        indices = random.argmax(1)
    negative = candidates[torch.arange(len(labels)), indices]
    return F.binary_cross_entropy_with_logits(torch.cat((positive, negative)),
                                              torch.cat((torch.ones_like(positive), torch.zeros_like(negative))))
