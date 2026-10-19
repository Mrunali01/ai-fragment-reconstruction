"""Image-only scoring and reconstruction shared by end-to-end evaluations."""

from PIL import Image
import torch

from jigsaw_ai.baseline_reconstruction import greedy_reconstruct, render_arrangement
from jigsaw_ai.boundary_model import BoundaryCNN, boundary_strips
from jigsaw_ai.edge_baseline import compatibility_costs
from jigsaw_ai.group_search import group_refine
from jigsaw_ai.improve_cnn import cost_dict, refine_swaps
from jigsaw_ai.tensor_data import image_tensor


@torch.inference_mode()
def score_pieces(pieces: list[Image.Image], model: BoundaryCNN | None = None) -> dict:
    """No model selects classical RGB MAE; a frozen model selects learned costs."""
    if len(pieces) != 25:
        raise ValueError("Expected 25 pieces")
    if model is None:
        return compatibility_costs(pieces)
    if model.training:
        raise ValueError("Set model.eval() before scoring")
    strips = boundary_strips(torch.stack([image_tensor(piece) for piece in pieces]))
    logits = model(strips)
    distances = (model.bias - logits) / model.log_scale.exp()
    distances /= distances[:, ~torch.eye(25, dtype=torch.bool)].mean().clamp_min(1e-8)
    if not torch.isfinite(distances).all():
        raise ValueError("Nonfinite CNN distances")
    return cost_dict(distances)


def assemble(costs: dict, search: str = "greedy") -> tuple[list[int], float]:
    if search not in ("greedy", "swaps", "groups", "beam"):
        raise ValueError("Unknown search method")
    order, cost = greedy_reconstruct(costs)
    if search != "greedy":
        order, cost = refine_swaps(order, costs)
    if search in ("groups", "beam"):
        order, cost = group_refine(order, costs, search)
    return order, cost


def reconstruct(pieces: list[Image.Image], model: BoundaryCNN | None = None,
                search: str = "greedy") -> dict:
    costs = score_pieces(pieces, model)
    order, cost = assemble(costs, search)
    return {"order_indices": order, "mean_border_cost": cost,
            "image": render_arrangement(pieces, order)}
