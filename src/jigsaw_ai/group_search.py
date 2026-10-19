"""Repair complete arrangements using group moves; CNN weights stay frozen."""

import numpy as np


def candidates(order: list[int]) -> np.ndarray:
    if len(order) != 25 or set(order) != set(range(25)):
        raise ValueError("Expected each of 25 pieces exactly once")
    grid = np.array(order).reshape(5, 5)
    result = [grid.copy()]
    for i in range(25):
        for j in range(i + 1, 25):
            candidate = grid.copy().reshape(25)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            result.append(candidate.reshape(5, 5))
    for axis in (0, 1):
        for line in range(5):
            for shift in range(1, 5):
                candidate = grid.copy()
                if axis == 0:
                    candidate[line, :] = np.roll(candidate[line, :], shift)
                else:
                    candidate[:, line] = np.roll(candidate[:, line], shift)
                result.append(candidate)
        for i in range(5):
            for j in range(i + 1, 5):
                candidate = grid.copy()
                if axis == 0:
                    candidate[[i, j], :] = candidate[[j, i], :]
                else:
                    candidate[:, [i, j]] = candidate[:, [j, i]]
                result.append(candidate)
    for dr in range(5):
        for dc in range(5):
            if dr or dc:
                result.append(np.roll(grid, (dr, dc), (0, 1)))
    blocks = [(r, c) for r in range(4) for c in range(4)]
    for index, (r, c) in enumerate(blocks):
        for rr, cc in blocks[index + 1:]:
            if abs(r - rr) < 2 and abs(c - cc) < 2:
                continue
            candidate = grid.copy()
            candidate[r:r + 2, c:c + 2] = grid[rr:rr + 2, cc:cc + 2]
            candidate[rr:rr + 2, cc:cc + 2] = grid[r:r + 2, c:c + 2]
            result.append(candidate)
    return np.unique(np.stack(result).reshape(-1, 25), axis=0)


def objective(orders: np.ndarray, costs: dict) -> np.ndarray:
    right, down = np.asarray(costs["RIGHT"]), np.asarray(costs["DOWN"])
    left = np.array([i for i in range(25) if i % 5 < 4])
    upper = np.arange(20)
    return (right[orders[:, left], orders[:, left + 1]].sum(1)
            + down[orders[:, upper], orders[:, upper + 5]].sum(1)) / 40


def group_refine(order: list[int], costs: dict, mode: str = "groups") -> tuple[list[int], float]:
    """Fixed search budgets: 20 improving rounds; optional width-4/depth-4 beam.

    No coordinates, puzzle IDs, reference images, or learned settings enter search.
    """
    if mode not in ("groups", "beam"):
        raise ValueError("Group search mode must be groups or beam")
    candidates(order)  # Validate before indexing any costs.
    best = np.array(order)
    score = float(objective(best[None], costs)[0])
    for _ in range(20):
        choices = candidates(best.tolist())
        values = objective(choices, costs)
        index = int(values.argmin())
        if values[index] >= score - 1e-12:
            break
        best, score = choices[index], float(values[index])
    if mode == "beam":
        beam = best[None]
        for _ in range(4):
            choices = np.unique(np.concatenate([candidates(row.tolist()) for row in beam]), axis=0)
            values = objective(choices, costs)
            indices = np.argsort(values, kind="stable")[:4]
            beam = choices[indices]
            if values[indices[0]] < score - 1e-12:
                best, score = beam[0], float(values[indices[0]])
    return best.tolist(), score
