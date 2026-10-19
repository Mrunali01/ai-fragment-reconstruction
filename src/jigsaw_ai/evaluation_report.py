"""Aggregate real predictions, paired uncertainty, and standalone report plots."""

import csv
from math import comb, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def aggregate(rows: list[dict]) -> dict:
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique puzzle results")
    n = len(rows)
    exact = sum(r["exact"] for r in rows)
    return {"puzzles": n, "exact_puzzles": exact, "exact_reconstruction_rate": exact / n,
            "fragment_position_accuracy": sum(r["position_correct"] for r in rows) / (25 * n),
            "assembly_neighbor_accuracy": sum(r["assembly_borders_correct"] for r in rows) / (40 * n),
            "neighbor_prediction_accuracy": sum(r["neighbor_correct"] for r in rows) / (80 * n),
            "pixel_position_accuracy": sum(r["pixel_position_correct"] for r in rows) / (25 * n),
            "pixel_exact_puzzles": sum(r["pixel_exact"] for r in rows),
            "pixel_exact_reconstruction_rate": sum(r["pixel_exact"] for r in rows) / n}


def appearance_metrics(order: list[int], true_order: list[int], fingerprints: list[str]) -> dict:
    """Supplement strict IDs with exact appearance equivalence for duplicate pieces."""
    if (len(order) != 25 or len(true_order) != 25 or len(fingerprints) != 25
            or set(order) != set(range(25)) or set(true_order) != set(range(25))):
        raise ValueError("Expected complete arrangements and 25 pixel fingerprints")
    correct = sum(fingerprints[a] == fingerprints[b] for a, b in zip(order, true_order))
    return {"pixel_position_correct": correct, "pixel_exact": correct == 25}


def wilson_interval(successes: int, total: int) -> list[float]:
    """Approximate 95% interval; the unit is an original puzzle, not a piece."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Invalid puzzle counts")
    z, p = 1.959963984540054, successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_comparison(before: list[dict], after: list[dict]) -> dict:
    a, b = {r["image_id"]: r for r in before}, {r["image_id"]: r for r in after}
    if not a or len(a) != len(before) or len(b) != len(after) or a.keys() != b.keys():
        raise ValueError("Paired comparison requires identical, unique puzzle IDs")
    ids = sorted(a)
    differences = np.array([int(b[i]["exact"]) - int(a[i]["exact"]) for i in ids])
    recovered = [i for i in ids if not a[i]["exact"] and b[i]["exact"]]
    regressed = [i for i in ids if a[i]["exact"] and not b[i]["exact"]]
    discordant = len(recovered) + len(regressed)
    tail = min(len(recovered), len(regressed))
    pvalue = min(1.0, 2 * sum(comb(discordant, k) for k in range(tail + 1)) / 2 ** discordant) if discordant else 1.0
    generator = np.random.default_rng(42)
    indices = generator.integers(0, len(ids), size=(10000, len(ids)))
    interval = np.quantile(differences[indices].mean(1), [0.025, 0.975]).tolist()
    return {"puzzles": len(ids), "recovered": recovered, "regressed": regressed,
            "net_exact_gain": len(recovered) - len(regressed), "exact_rate_difference": float(differences.mean()),
            "paired_bootstrap_95_percent_interval": interval, "bootstrap_resamples": 10000,
            "bootstrap_seed": 42, "two_sided_discordant_binomial_p": pvalue,
            "interpretation": "Exploratory paired puzzle-level uncertainty; no multiple-comparison correction or method selection."}


def save_report(output: Path, cohorts: dict, calibration: dict, comparisons: dict, failures: list[dict]) -> None:
    fields = ["cohort", "method", "puzzles", "exact_puzzles", "exact_reconstruction_rate",
              "fragment_position_accuracy", "assembly_neighbor_accuracy", "neighbor_prediction_accuracy",
              "pixel_position_accuracy", "pixel_exact_puzzles", "pixel_exact_reconstruction_rate"]
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for cohort, report in cohorts.items():
            for method, metrics in report["methods"].items():
                writer.writerow({"cohort": cohort, "method": method, **metrics})
    fresh = cohorts["fresh_observed_holdout"]
    methods = list(fresh["methods"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    rates = [fresh["methods"][m]["exact_reconstruction_rate"] for m in methods]
    intervals = [wilson_interval(fresh["methods"][m]["exact_puzzles"], fresh["methods"][m]["puzzles"]) for m in methods]
    axes[0].bar(range(len(methods)), rates, color=["#64748b"] * 3 + ["#2563eb"] * 3)
    axes[0].errorbar(range(len(methods)), rates,
                     yerr=[[p - interval[0] for p, interval in zip(rates, intervals)],
                           [interval[1] - p for p, interval in zip(rates, intervals)]], fmt="none", color="black", capsize=3)
    axes[0].set_title("Exact puzzles; approximate 95% Wilson intervals")
    axes[1].bar(range(len(methods)), [fresh["methods"][m]["fragment_position_accuracy"] for m in methods],
                color=["#64748b"] * 3 + ["#2563eb"] * 3)
    axes[1].set_title("Absolute piece positions")
    for axis in axes:
        axis.set_xticks(range(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=8)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Accuracy")
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)
    lines = ["# Phase 8: end-to-end evaluation", "",
             "All methods use identical shuffled pieces within each cohort. No retraining or test-driven tuning.", "",
             "The 50-image cohort was fresh when acquired before its first evaluation; it is now observed.", ""]
    for cohort, report in cohorts.items():
        lines += [f"## {cohort}", "", "| Method | Exact | Positions | Assembly borders | Neighbor ranking |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for method, metrics in report["methods"].items():
            lines.append(f"| {method} | {metrics['exact_puzzles']}/{metrics['puzzles']} | {metrics['fragment_position_accuracy']:.2%} | {metrics['assembly_neighbor_accuracy']:.2%} | {metrics['neighbor_prediction_accuracy']:.2%} |")
        lines += ["", "Pair classification: all 600 ordered nonself pairs per puzzle, 80 positives.", ""]
        for scorer, metrics in report["pairwise"].items():
            lines.append(f"- {scorer}: precision {metrics['precision']:.2%}, recall {metrics['recall']:.2%}, F1 {metrics['f1']:.2%}; TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}.")
        lines.append("")
        lines += ["Supplementary exact-pixel equivalence (does not replace strict IDs):", ""]
        for method, metrics in report["methods"].items():
            lines.append(f"- {method}: {metrics['pixel_exact_puzzles']}/{metrics['puzzles']} pixel-exact puzzles; {metrics['pixel_position_accuracy']:.2%} appearance-correct piece positions.")
        lines.append("")
    lines += ["## Paired fresh-cohort comparisons", ""]
    for name, result in comparisons.items():
        low, high = result["paired_bootstrap_95_percent_interval"]
        lines.append(f"- {name}: {len(result['recovered'])} recovered, {len(result['regressed'])} regressed; exact-rate difference {result['exact_rate_difference']:+.1%}, exploratory paired bootstrap interval [{low:+.1%}, {high:+.1%}], discordant-binomial p={result['two_sided_discordant_binomial_p']:.4f}.")
    lines += ["", "## Calibration and limitations", "",
              "Pair thresholds were fit only on the original validation originals, then applied unchanged. They do not affect assembly.",
              f"Classical pair threshold: {calibration['classical']['threshold']:.12g}; CNN pair threshold: {calibration['cnn']['threshold']:.12g}.",
              "CNN pair cost is the minimum normalized learned distance over four directions, not a calibrated probability.",
              "Always predicting non-neighbor gives 86.67% pair accuracy but zero positive F1. Exact reconstruction remains the primary task.",
              "Piece positions and borders within a puzzle are correlated; uncertainty is computed using puzzles as the units.",
              "Comparisons are descriptive/exploratory, not evidence of universal superiority or a new untouched final benchmark.", "",
              "## Primary CNN-beam failures", ""]
    for failure in failures:
        duplicates = failure["pixel_identical_piece_groups"]
        lines.append(f"- {failure['image_id']}: {failure['position_correct']}/25 strict positions, {failure['pixel_position_correct']}/25 appearance positions, {failure['assembly_borders_correct']}/40 borders; {failure['objective_diagnosis']}. Pixel-identical groups: {duplicates}.")
    lines += ["", "Ground truth enters metric and failure analysis only, after prediction. No solver receives coordinates or the reference image.",
              "Every stored arrangement uses all 25 pieces exactly once. Supported puzzles remain upright 5 x 5 square fragments.",
              "Pixel equivalence is supplementary: distinct piece IDs with identical RGB pixels cannot be distinguished from appearance alone. Primary strict metrics are unchanged.",
              "Pairwise costs and labels are stored in allow_pickle=False-readable NPZ files for independent recalculation.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
