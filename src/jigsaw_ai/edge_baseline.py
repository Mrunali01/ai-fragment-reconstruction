"""Mean absolute RGB differences between touching edges (lower is better)."""

from PIL import Image

DIRECTIONS = ("LEFT", "RIGHT", "UP", "DOWN")
OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}


def edges(image: Image.Image) -> dict[str, bytes]:
    if image.mode != "RGB" or image.size != (64, 64):
        raise ValueError("Expected an RGB 64x64 piece")
    return {
        "LEFT": image.crop((0, 0, 1, 64)).tobytes(),
        "RIGHT": image.crop((63, 0, 64, 64)).tobytes(),
        "UP": image.crop((0, 0, 64, 1)).tobytes(),
        "DOWN": image.crop((0, 63, 64, 64)).tobytes(),
    }


def edge_difference(first: Image.Image, second: Image.Image, direction: str) -> float:
    """Direction describes the second image relative to the first."""
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}")
    a, b = edges(first)[direction], edges(second)[OPPOSITE[direction]]
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def compatibility_costs(pieces: list[Image.Image]) -> dict[str, list[list[float]]]:
    if len(pieces) != 25:
        raise ValueError("Expected exactly 25 pieces")
    vectors = [edges(piece) for piece in pieces]
    costs = {d: [[float("inf")] * 25 for _ in range(25)] for d in DIRECTIONS}
    for i in range(25):
        for j in range(25):
            if i == j:
                continue
            for direction in ("RIGHT", "DOWN"):
                a, b = vectors[i][direction], vectors[j][OPPOSITE[direction]]
                value = sum(abs(x - y) for x, y in zip(a, b)) / len(a)
                costs[direction][i][j] = value
                costs[OPPOSITE[direction]][j][i] = value
    return costs
