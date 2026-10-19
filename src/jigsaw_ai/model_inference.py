"""Score all ordered pairs by encoding each puzzle piece only once."""

from PIL import Image
import torch

from jigsaw_ai.pairs import DIRECTION_IDS
from jigsaw_ai.siamese import SiameseCNN
from jigsaw_ai.tensor_data import image_tensor


@torch.inference_mode()
def score_puzzle(model: SiameseCNN, pieces: list[Image.Image], batch_size: int = 128) -> dict[str, torch.Tensor]:
    if model.training:
        raise ValueError("Set model.eval() before inference")
    if len(pieces) != 25 or batch_size < 1:
        raise ValueError("Expected 25 pieces and a positive batch size")
    parameter = next(model.parameters())
    inputs = torch.stack([image_tensor(p) for p in pieces]).to(device=parameter.device, dtype=parameter.dtype)
    embeddings = model.encoder(inputs)
    pairs = [(i, j) for i in range(25) for j in range(25) if i != j]
    compatibility = torch.zeros(25, 25)
    conditional_direction = torch.zeros(25, 25, 4)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        first = [i for i, _ in batch]
        second = [j for _, j in batch]
        features = model.pair_head(torch.cat((embeddings[first], embeddings[second]), dim=1))
        binary = model.compatibility_head(features).squeeze(-1).sigmoid().cpu()
        directional = model.direction_head(features).softmax(1).cpu()
        compatibility[first, second] = binary
        conditional_direction[first, second] = directional
    return {"compatibility": compatibility, "conditional_direction": conditional_direction,
            "joint_direction": compatibility.unsqueeze(-1) * conditional_direction}


def neighbor_queries(scores: dict[str, torch.Tensor], positions: list[tuple[int, int]]) -> list[dict]:
    if len(positions) != 25 or set(positions) != {(r, c) for r in range(5) for c in range(5)}:
        raise ValueError("Expected complete 5x5 ground truth")
    offsets = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (-1, 0), "DOWN": (1, 0)}
    lookup = {position: i for i, position in enumerate(positions)}
    queries = []
    for i, (row, column) in enumerate(positions):
        for direction, (dr, dc) in offsets.items():
            target = lookup.get((row + dr, column + dc))
            if target is not None:
                candidates = scores["joint_direction"][i, :, DIRECTION_IDS[direction]].clone()
                candidates[i] = -float("inf")
                prediction = int(candidates.argmax())
                queries.append({"first_index": i, "direction": direction, "target_index": target,
                                "prediction_index": prediction, "predicted_joint_score": float(candidates[prediction]),
                                "target_joint_score": float(candidates[target])})
    return queries


def neighbor_counts(scores: dict[str, torch.Tensor], positions: list[tuple[int, int]]) -> tuple[int, int]:
    queries = neighbor_queries(scores, positions)
    return sum(q["prediction_index"] == q["target_index"] for q in queries), len(queries)
