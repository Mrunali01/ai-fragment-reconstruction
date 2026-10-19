"""Greedy row-major reconstruction using edge costs, never position labels."""

from PIL import Image

from jigsaw_ai.edge_baseline import compatibility_costs


def arrangement_cost(order: list[int], costs: dict) -> float:
    if len(order) != 25 or set(order) != set(range(25)):
        raise ValueError("Arrangement must use each of the 25 pieces exactly once")
    total = 0.0
    for index, piece in enumerate(order):
        row, column = divmod(index, 5)
        if column:
            total += costs["RIGHT"][order[index - 1]][piece]
        if row:
            total += costs["DOWN"][order[index - 5]][piece]
    return total / 40


def greedy_reconstruct(costs: dict) -> tuple[list[int], float]:
    """Try every top-left anchor; select the lowest mean cost over 40 borders."""
    for direction in ("LEFT", "RIGHT", "UP", "DOWN"):
        matrix = costs[direction]
        if len(matrix) != 25 or any(len(row) != 25 for row in matrix):
            raise ValueError("Expected 25x25 directional cost matrices")
    candidates = []
    for anchor in range(25):
        order, unused = [anchor], set(range(25)) - {anchor}
        for index in range(1, 25):
            row, column = divmod(index, 5)
            def local_cost(piece):
                value = costs["RIGHT"][order[index - 1]][piece] if column else 0.0
                if row:
                    value += costs["DOWN"][order[index - 5]][piece]
                return value, piece
            chosen = min(unused, key=local_cost)
            order.append(chosen)
            unused.remove(chosen)
        candidates.append((arrangement_cost(order, costs), order))
    cost, order = min(candidates)
    return order, cost


def reconstruct(pieces: list[Image.Image]) -> tuple[list[int], float]:
    return greedy_reconstruct(compatibility_costs(pieces))


def render_arrangement(pieces: list[Image.Image], order: list[int]) -> Image.Image:
    if len(pieces) != 25 or len(order) != 25 or set(order) != set(range(25)):
        raise ValueError("Expected 25 pieces and a complete arrangement")
    canvas = Image.new("RGB", (320, 320))
    for index, piece_id in enumerate(order):
        piece = pieces[piece_id]
        if piece.mode != "RGB" or piece.size != (64, 64):
            raise ValueError("Expected RGB 64x64 pieces")
        canvas.paste(piece, ((index % 5) * 64, (index // 5) * 64))
    return canvas
