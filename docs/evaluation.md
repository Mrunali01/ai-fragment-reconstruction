# Phase 8: end-to-end ML evaluation

Phase 8 connects shuffled fragment pixels to complete reconstructed images and
reports compatibility, neighbor, position, and exact-puzzle metrics. It compares
classical and neural scoring under the same search methods. No model was trained,
no new dataset was selected, and no settings were tuned on observed test results.
FastAPI, storage, and frontend work remain later phases.

## Run and deliverables

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_pipeline
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

The default output is `artifacts/phase8/final`. It must be new/empty. Supply
`--output` to reproduce into a different directory. `--development`, `--holdout`,
`--checkpoint`, `--selection`, and `--protocol` accept explicit input paths.
Defaults reference the already acquired 500-image development collection and
the separate 50-image holdout from the search experiment; this command does not
silently download or assume another dataset. Missing files produce an error.

Artifacts:

- `config.json`: frozen methods, dataset/metadata fingerprints, checkpoint,
  selection/protocol hashes, core source-file hashes, and Torch version/CPU threads.
- `calibration.json`: the two validation-only pair thresholds and their provenance.
- `summary.json`, `report.md`, and `metrics.csv`: all metrics, comparisons, and limitations.
- `comparison.png`: standalone accuracy plot with approximate puzzle-level
  95% Wilson intervals for exact reconstruction rates.
- `validation/` and `fresh_observed_holdout/`: all six per-puzzle method reports,
  cohort summaries, primary CNN-beam predictions for every puzzle, failure
  diagnostics, and failure images containing target/input/classical/CNN panels.
- `pair_predictions.npz` in each cohort: puzzle IDs, all ground-truth pair labels,
  and classical/CNN costs. Load with `np.load(path, allow_pickle=False)` to
  independently recalculate the confusion matrices.

An initial diagnostic report is retained under `artifacts/phase8/v1`; the final
report includes supplementary visual equivalence and source fingerprints.
Generated data, checkpoints, and reports remain ignored by Git.

## Pipeline walkthrough

`pipeline.score_pieces(pieces, model)` accepts 25 RGB 64 x 64 PIL pieces. With
`model=None`, it returns classical touching-edge RGB mean absolute differences;
with the frozen boundary CNN, it returns the learned directional distances
normalized by the mean over all nonself candidates within the puzzle. This
normalization uses only input pixels and predictions.

`pipeline.assemble(costs, search)` turns scores into a permutation of piece
indices, using greedy placement, swaps, group moves, or beam search.
`pipeline.reconstruct(pieces, model, search)` connects those operations and
returns indices, mean border cost, and the reconstructed RGB image. Neither
function receives original coordinates or a reference image.

`evaluate_pipeline` validates input provenance and verifies each saved puzzle.
It then makes all six image-only predictions before using original positions
for metrics. The two scorers are paired with greedy, swaps, and beam search,
so scoring and search effects can be assessed separately. Beam parameters
remain the frozen Phase 7 improvement settings; the baseline beam control is
not separately optimized. Primary method remains CNN beam, selected earlier on
validation. No fresh-cohort winner is selected by this command.

For pair classification, each scorer's threshold maximizes F1 on the original
75 validation images only. The threshold is then applied unchanged to the
50-image cohort. The pair thresholds do not affect reconstruction: assembly
uses directional costs directly. The classical threshold reproduces Phase 4's
12.807291666666666. The CNN threshold is 0.19765564799308777 in its normalized
distance units. These values and raw costs are not calibrated probabilities.

## Metrics and denominators

- **Pair compatibility:** all 600 ordered nonself pairs per puzzle; 80 true
  neighbors and 520 non-neighbors. Pair cost is the minimum directional cost.
  Report TP/FP/TN/FN, precision, recall, F1, and accuracy. Predicting every pair
  as negative gives 86.67% accuracy and zero positive F1, so accuracy alone is misleading.
- **Neighbor prediction accuracy:** 80 true directional queries per puzzle,
  with the true neighbor ranked against all 24 other pieces.
- **Fragment position accuracy:** exact original piece ID in its original
  row and column, divided by 25 per puzzle. Global shifts remain errors.
- **Assembly neighbor accuracy:** correct RIGHT/DOWN relationships among the
  40 borders in the completed arrangement.
- **Exact reconstruction rate:** all 25 original piece IDs in their original
  positions, divided by the number of puzzles.
- **Supplementary pixel equivalence:** position is visually correct when the
  predicted piece has exactly the same RGB pixels as the true piece at that
  location, even if its ID differs. Pixel-exact reconstruction requires all
  25 positions to match. It does not replace or alter the primary strict metrics.

The supplementary metric was added after discovering pixel-identical fragments
during failure analysis. It is descriptive, not a new selection criterion. In
validation, CNN swaps has 64/75 strict exact puzzles but 65/75 pixel-exact puzzles.
For the primary CNN beam, strict and pixel-exact counts agree: 68/75 validation
and 45/50 on the 50-image cohort. Thus the new diagnostic does not inflate the
main result. Matching costs from classical and CNN scorers have different units;
do not compare their magnitudes directly.

## Actual results (October 19, 2026)

The original 75 validation puzzles and the separate 50-image cohort were evaluated
independently. Counts were not pooled across cohorts.

| Scoring and search | Validation exact | 50-image exact | 50-image positions | 50-image assembled borders |
| --- | ---: | ---: | ---: | ---: |
| Classical + greedy | 59/75 | 37/50 (74%) | 82.64% | 94.45% |
| Classical + swaps | 59/75 | 40/50 (80%) | 83.52% | 95.40% |
| Classical + beam | 66/75 | 43/50 (86%) | 86.64% | 96.05% |
| CNN + greedy | 63/75 | 41/50 (82%) | 86.48% | 96.15% |
| CNN + swaps | 64/75 | 42/50 (84%) | 86.80% | 96.30% |
| CNN + beam | 68/75 | 45/50 (90%) | 90.00% | 97.05% |

Neighbor accuracy is unaffected by search: on the 50-image cohort it is 96.25%
for classical scoring and 96.80% for the CNN. CNN-beam minus classical-greedy
combines scoring and search changes. The same-beam comparison is the more direct
scorer comparison: 45/50 versus 43/50, a modest two-puzzle difference.

Pair compatibility on 30,000 ordered pairs from the 50-image cohort (4,000 positive):

| Scorer | Precision | Recall | F1 | TP / FP / TN / FN |
| --- | ---: | ---: | ---: | --- |
| Classical | 46.33% | 81.80% | 59.16% | 3272 / 3790 / 22210 / 728 |
| CNN | 57.23% | 83.35% | 67.86% | 3334 / 2492 / 23508 / 666 |

Validation pair F1 is 62.28% classical and 68.67% CNN over 45,000 pairs. Both
validation and 50-image reports include all confusion-matrix counts, thresholds,
and raw predictions for independent verification.

Historical context is a different cohort: on the original 75-image test set,
the old whole-piece CNN solved 0/75, classical greedy solved 63/75, and the
boundary CNN with swaps solved 68/75. Those numbers remain unchanged and must
not be compared directly with the new 50-image counts.

## Uncertainty and overfitting limitations

The 50 images were new when the prior search improvement protocol was frozen
and first evaluated. They are now observed and are reused here only for
descriptive evaluation, not model/search selection. This phase is not another
untouched test. Future improvement experiments need a new independent final
cohort rather than repeatedly optimizing these images.

The original 500 IDs, source URLs, file/pixel hashes, known photographer profiles,
and close prepared-image fingerprints were excluded during acquisition. Auditing
also found 50 distinct known photographer profiles within the new 50 images.
This reduces known overlap but does not prove complete semantic independence.
The bounded Open Images CSV prefix, license/access filtering, and small size
limit population representativeness.

Uncertainty uses original puzzles as the units, because piece and border outcomes
within a puzzle are correlated. The report includes 10,000 paired bootstrap
resamples (seed 42), approximate Wilson intervals, and two-sided exact binomial
tests over discordant puzzle outcomes. Comparisons are exploratory, with no
multiple-comparison correction and no method selection from these statistics.

- CNN beam versus CNN swaps: five recoveries, two regressions; +6 percentage
  points, paired bootstrap interval **[-4, +16] points**, discordant-binomial p=0.453125.
- CNN beam versus classical beam: two recoveries, zero regressions; +4 points,
  interval **[0, +10] points**, p=0.5.

These small margins do not establish reliable universal superiority. The point
estimates favor CNN beam on this cohort, while the intervals show substantial
uncertainty. No additional training, augmentation, threshold retuning on test,
new architecture, or search-budget change was introduced in Phase 8.

## Remaining primary failures

All five fresh-cohort CNN-beam failures assign lower cost to the wrong arrangement
than to ground truth. This demonstrates an objective/scoring limitation, not
just insufficient search depth. Broader search alone cannot guarantee correction.

| Image ID | Description | Strict positions | Correct borders | Observed issue |
| --- | --- | ---: | ---: | --- |
| 9348880c9a8cc257 | Person on grass | 0/25 | 35/40 | Shifted arrangement preferred by objective |
| d31b3f1c2cb862c7 | Chocolate wrapper on pale background | 0/25 | 30/40 | Background matches favor broken placement |
| 2edfa05fc6822135 | Line chart | 0/25 | 27/40 | Repeated grid/text/background, 50/80 correct neighbor queries |
| 1ad9b1deae45d64c | Patterned fabric on white background | 0/25 | 35/40 | Shifted arrangement preferred by objective |
| dc0d956250c5ce81 | Product box on white background | 0/25 | 14/40 | Nine pixel-identical pieces, 42/80 correct neighbor queries |

Nine identical pieces are inherently indistinguishable by their RGB appearance;
the original fragment IDs cannot always be recovered uniquely. The current box
prediction also misplaces nonidentical pieces, so this case is not merely an ID
bookkeeping error. The per-puzzle report records supplementary visual correctness
and duplicate groups rather than dismissing all errors as unavoidable.

Border-aware/context scoring remains a possible future experiment, not an
implemented fix or a promised improvement. Supported inputs remain complete
5 x 5 puzzles with 25 upright RGB 64 x 64 square pieces. No GPU validation was
performed; evaluation and tests run on the available CPU environment.

## Tests and checks

Four new tests exercise paired puzzle-level statistics, image-only end-to-end
reconstruction and unchanged weights, exact appearance equivalence, and a full
synthetic validation-to-holdout report. Tests independently recalculate stored
NPZ confusion counts and aggregate position/exact metrics, validate all
permutations, and check overwrite, source overlap, and development-image reuse
rejection. The controlled dataset/checkpoint is test-only; its scores are not
reported as real project performance.

The full suite passed **91 tests in 64.43 seconds**; the focused new suite passed
**4 tests in 9.87 seconds**. `pip check` and `git diff --check` passed. Real
evaluation saved all 125 primary reconstructed images plus all primary failure
previews and raw predictions. No new dependency was added.

A complete rerun into `artifacts/phase8/reproduction` produced **161 files with
identical SHA-256 hashes**, including reports, NPZ predictions, plots, and images.
All 50 primary fresh-cohort arrangements also matched the prior frozen search
audit exactly. Classical validation calibration and pair metrics reproduced
Phase 4 exactly. `artifacts/phase8/final/verification.json` records these checks.
