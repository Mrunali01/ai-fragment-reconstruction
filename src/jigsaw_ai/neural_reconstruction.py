"""Turn frozen CNN scores into a complete arrangement, without ground truth."""

from PIL import Image
import torch

from jigsaw_ai.baseline_reconstruction import greedy_reconstruct
from jigsaw_ai.model_inference import score_puzzle
from jigsaw_ai.pairs import DIRECTION_IDS
from jigsaw_ai.siamese import SiameseCNN


def directional_costs(scores: dict[str, torch.Tensor]) -> dict[str, list[list[float]]]:
    """Lower cost means a higher ordered compatibility-and-direction score."""
    joint = scores["joint_direction"].detach().cpu()
    if joint.shape != (25, 25, 4):
        raise ValueError("Expected joint scores with shape 25x25x4")
    if not torch.isfinite(joint).all() or not ((joint >= 0) & (joint <= 1)).all():
        raise ValueError("Joint scores must be finite and within 0..1")
    costs = {}
    for direction, index in DIRECTION_IDS.items():
        matrix = 1.0 - joint[:, :, index]
        matrix.fill_diagonal_(float("inf"))
        costs[direction] = matrix.tolist()
    return costs


def reconstruct(model: SiameseCNN, pieces: list[Image.Image]) -> tuple[list[int], float]:
    """Return 25 shuffled piece indices in row-major order and mean border cost."""
    return greedy_reconstruct(directional_costs(score_puzzle(model, pieces)))
