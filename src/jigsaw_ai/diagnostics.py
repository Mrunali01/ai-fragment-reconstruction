"""Relative edge evidence from actual solver costs, never calibrated accuracy."""
import numpy as np


def compatibility_report(costs: dict, order: list[int], scorer: str) -> dict:
    edges = []
    for position, piece in enumerate(order):
        row, column = divmod(position, 5)
        neighbors = []
        if column < 4:
            neighbors.append(("RIGHT", order[position + 1]))
        if row < 4:
            neighbors.append(("DOWN", order[position + 5]))
        for direction, candidate in neighbors:
            values = np.asarray(costs[direction][piece], dtype=float)
            alternatives = [(float(values[index]), index) for index in range(25) if index not in (piece, candidate)]
            alternatives.sort()
            chosen = float(values[candidate])
            other_best = alternatives[0][0]
            margin = (other_best - chosen) / (abs(other_best) + abs(chosen) + 1e-12)
            rank = 1 + sum(value < chosen for value, index in alternatives)
            edges.append({"piece": piece, "neighbor": candidate, "direction": direction,
                          "cost": chosen, "rank": rank, "margin": margin,
                          "alternatives": [{"piece": index, "cost": value} for value, index in alternatives[:3]]})
    return {"schema_version": 1, "scorer": scorer, "edge_count": len(edges), "edges": edges,
            "mean_border_cost": sum(edge["cost"] for edge in edges) / len(edges),
            "top_ranked_border_fraction": sum(edge["rank"] == 1 for edge in edges) / len(edges),
            "mean_margin": sum(edge["margin"] for edge in edges) / len(edges),
            "definition": "Rank compares a chosen neighbor with the other 23 non-self candidates; ties share rank. Margin=(best alternative cost-chosen cost)/(absolute costs sum+1e-12). Lower cost is better.",
            "warning": "Relative matching evidence, not calibrated confidence or reconstruction accuracy. Costs are not comparable across CNN and classical scorers."}
