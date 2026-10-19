# Phase 4: classical edge-matching baseline

This phase implements an interpretable pixel-based solver. It uses no neural
network, training images, original positions, shuffle seeds, or target image to
choose an arrangement. Labels and targets are read separately for evaluation.
No dependencies were added.

## Edge compatibility

For a proposed RIGHT neighbor, compare the first piece's rightmost pixel column
with the second piece's leftmost column. For DOWN, compare the first piece's
bottom row with the second piece's top row. LEFT and UP work analogously.

The cost is the mean absolute channel difference over 64 pixels x 3 RGB channels:

`cost = sum(abs(edge_a[channel] - edge_b[channel])) / 192`

Costs range from 0 to 255. Lower means more similar touching edges. Self-matches
are excluded with infinite cost. Opposite directions are symmetric when the pair
order is reversed. Similarity is not a calibrated probability of correctness.

## Reconstruction algorithm

The solver knows only that there are 25 unrotated 64x64 RGB pieces in a 5x5 grid.

1. Try each piece as the top-left anchor (25 independent greedy attempts).
2. Fill cells from left to right, top to bottom.
3. At each cell, choose the unused piece minimizing the sum of costs against the
   already placed left and upper neighbors, when those neighbors exist.
4. Compute each completed arrangement's mean cost over its 40 internal borders.
5. Return the lowest-cost arrangement. Ties use shuffled piece indices.

Every piece is used exactly once. Trying all anchors avoids choosing one arbitrary
first piece, but does not make greedy search globally optimal. Once placed, a
piece cannot move during an attempt. No ground-truth coordinates enter these
steps. Source filenames describe shuffled order, not the correct arrangement.

`reconstruct(pieces)` is the image-only API. `compatibility_costs(pieces)` and
`greedy_reconstruct(costs)` expose the two stages separately for evaluation and
reuse. `render_arrangement()` displays the resulting permutation.

## Evaluation and metric definitions

We fix this algorithm before inspecting test results. Validation chooses only a
binary pair-compatibility threshold, maximizing validation F1. Equal-cost pairs
are grouped; equal F1 scores choose the lower threshold. The test command requires
the saved validation calibration and rejects a different dataset or method.
The threshold does not affect the greedy solver or directional neighbor ranking.

- **Pair precision / recall / F1 / accuracy:** score all 600 ordered non-self
  pairs per puzzle. A pair's cost is its minimum over the four directions, because
  this binary label asks whether it touches in any direction. Predict compatible
  when cost is at most the validation-selected threshold. Report actual TP, FP,
  TN, FN, precision, recall, F1, and accuracy.
- **Neighbor prediction accuracy:** for each of the 80 true directed relationships
  per puzzle, choose the lowest-cost candidate among the other 24 pieces in that
  direction. Count the correct choices. Queries outside the true grid are excluded
  by evaluation only; the scorer itself has no boundary labels.
- **Fragment position accuracy:** correct absolute grid positions divided by 25
  per puzzle. No translation, rotation, or reflection alignment is allowed.
- **Assembly neighbor accuracy:** correct oriented internal borders divided by
  the 40 borders in the predicted arrangement.
- **Exact reconstruction rate:** fraction of puzzles with all 25 absolute
  positions correct. A visually plausible alternative is still incorrect under
  the recorded ground truth.

Aggregates pool the relevant counts across puzzles; every puzzle has the same
number of pieces and borders. Pair evaluation uses all negatives, rather than
Phase 3's balanced negative sample: 80 positive and 520 negative pairs per puzzle.
Do not compare pair accuracy across those different class distributions without
accounting for the imbalance. There is no measured CNN comparison yet.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_baseline --split validation
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_baseline --split test
```

Options include `--source`, `--output`, and `--calibration`. Default outputs are
`data/baseline/validation` and `data/baseline/test`; the test command defaults to
`data/baseline/validation/calibration.json` for calibration. Only validation/test
evaluation is exposed by this CLI. Nothing is tuned on the test results.

## Local artifacts

Each split output directory contains:

- `config.json`: frozen method, input hashes, versions, calibration hash for test.
- `summary.json`: aggregate actual metrics and threshold.
- `results.json`: per-puzzle predictions, edge costs, and evaluation counts.
- `predictions/<image_id>.png`: every predicted arrangement.
- `previews/01.png` to `03.png`: the first three indexed puzzles, with target,
  shuffled input, and automatic prediction. Selection does not depend on accuracy.
- `failure_analysis.json`: failed puzzle IDs, lowest position accuracies, and
  algorithmic limitations.
- `calibration.json`: validation only; selected threshold and method provenance.

All artifacts are Git-ignored. Reruns with the same settings overwrite known
outputs; changed input hashes require a fresh output directory. Nonempty unrelated
output directories and directories overlapping the puzzle source are rejected.
Run only one evaluation per output directory at a time.

## Tests and limitations

Tests cover a hand-calculated RGB difference, directional symmetry, self-match
exclusion, exact recovery of a controlled border-coded puzzle, deterministic ties,
one-use-per-piece constraints, exact/position/border metrics, threshold tie
handling, validation-only calibration, and reproducible evaluation exports.

Uniform sky, repeated textures, dark areas, and matching colors at unrelated
edges are ambiguous. Real neighboring pixels need not have equal colors.
Greedy mistakes can cascade, and low global cost can still describe a wrong
arrangement. Center cropping and resizing define the target; this baseline does
not restore content removed during preprocessing. Unknown source orientations
remain as documented in Phase 2.

## Actual results (October 16, 2026)

The method above was fixed before test evaluation. The validation threshold is
12.807291666666666 RGB difference units and is reused unchanged for test.

| Metric | Validation (75 puzzles) | Test (75 puzzles) |
| --- | --- | --- |
| Pair examples | 45,000 | 45,000 |
| Pair precision | 52.34% | 51.11% |
| Pair recall | 76.90% | 77.63% |
| Pair F1 | 62.28% | 61.64% |
| Pair accuracy | 87.58% | 87.12% |
| Directional neighbor prediction accuracy | 95.85% | 97.70% |
| Absolute fragment position accuracy | 83.36% | 88.32% |
| Assembly neighbor accuracy | 93.33% | 96.83% |
| Exact puzzles | 59 / 75 | 63 / 75 |
| Exact reconstruction rate | 78.67% | 84.00% |

Test confusion counts: TP 4,658; FP 4,456; TN 34,544; FN 1,342. An always-negative
pair classifier would score 86.67% accuracy on this distribution, so pair accuracy
alone overstates binary-classification quality. Directional neighbor ranking and
full reconstruction answer different questions and are reported separately.

There were 16 validation failures and 12 test failures. Six deterministic preview
panels (first three puzzles in each split) were inspected, along with these actual
test failures:

- `9aa39e3502c6089e`: light ring on a dark background. The prediction is shifted,
  preserving 35/40 internal relationships but matching 0/25 absolute positions.
  Local neighbor ranking is 100% on this puzzle. Similar outer-background edges
  let the global cost prefer a shifted arrangement despite strong local ranking.
- `3d8e8a9add2c2e53`: diagram with white regions. The prediction misplaces diagram
  blocks; position accuracy is 56%, assembly neighbor accuracy 50%, and directional
  neighbor accuracy 67.5%. Repeated white backgrounds give weak edge evidence.

These are observations from this small pilot collection, not a general accuracy
guarantee or proof that a future CNN will outperform the baseline. No algorithm
or threshold was adjusted after inspecting the test results.

The full automated suite passed **41 tests**, with no failures. `pip check`
reported no broken requirements. Both evaluation commands completed successfully;
all 150 predicted arrangements contain each of their 25 pieces exactly once.
Generated evaluation artifacts occupy 28,513,629 bytes (about 28.5 MB).
A full validation-and-test rerun reproduced every evaluation artifact byte for
byte, checked by SHA256. Git checks confirmed outputs are ignored and no remote
is configured. No commit or push was performed.
