# Phase 7: frozen CNN reconstruction

The approved Phase 7 connects the Phase 6 checkpoint to the existing greedy
assembler. It reconstructs all validation and test puzzles and saves actual
predictions. No weights, hyperparameters, or compatibility thresholds are changed.
The current neural approach performs substantially worse than the classical baseline.

## Run

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_reconstruction --split validation
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_reconstruction --split test
.\.venv\Scripts\python.exe -m pytest -q
```

Defaults use `data/puzzles`, `artifacts/training/siamese_cpu_v1/best.pt`, and the
Phase 4 reports in `data/baseline`. Outputs go to
`artifacts/reconstruction/{validation,test}`. Use `--source`, `--checkpoint`,
`--output`, `--validation`, and `--baseline-root` to supply other locations.
Test requires a completed validation run using the same method, checkpoint,
and dataset. There is no binary threshold to calibrate for assembly: candidates
are ranked by their directional scores. The method was fixed before test evaluation.

## How it works

1. Load the frozen checkpoint in evaluation mode. Encode each of the 25 pieces
   once; score all 600 ordered, distinct pairs using the trained heads.
2. For piece B relative to A in direction D, use the score
   `sigmoid(compatibility_logit) * softmax(direction_logits)[D]`.
   Convert it to a cost with `1 - score`, so lower is better. Reject nonfinite,
   incorrectly shaped, and out-of-range scores. Set self-pair costs to infinity.
3. Reuse `greedy_reconstruct`: try each piece as the top-left anchor. Fill the
   grid in row-major order, choosing the unused piece with the smallest sum of
   costs to the already placed left and upper neighbors. Use the left piece's
   RIGHT score and upper piece's DOWN score. An explicit unused set guarantees
   every piece appears exactly once, with no empty cells.
4. Choose the completed candidate with the lowest mean cost over its 40 borders.
   Candidate ties use the lowest shuffled piece index; completed-arrangement ties
   use lexicographic order. No random operation occurs during solving.
5. Render the selected piece order on a 320 x 320 RGB canvas. Only after solving,
   use the stored original positions to calculate evaluation metrics.

`neural_reconstruction.reconstruct(model, pieces)` accepts only an evaluation-mode
model and 25 RGB 64 x 64 PIL images. It returns 25 indices into the supplied list
in row-major order, plus mean border cost. It needs neither original positions
nor the intact image. `render_arrangement` creates the corresponding image.
The evaluation CLI additionally needs the generated puzzle index and metadata
to check dataset integrity and measure accuracy.

This deliberately uses the same assembly algorithm as Phase 4. Only the scoring
function changes. Scores in opposite directions can disagree; this first solver
does not combine the two directions or backtrack. Its 25 candidate arrangements
are a small subset of all possible arrangements, so it cannot guarantee the
globally optimal arrangement or even a low-cost arrangement that is correct.

## Outputs and safeguards

Each split produces:

- `config.json`: frozen method, checkpoint SHA-256, index and metadata hashes,
  validation config hash for test, and baseline summary hash.
- `results.json`: every puzzle's ordered piece IDs, border cost, metric counts,
  and rates. These IDs map directly to that puzzle's saved pieces.
- `summary.json`: aggregate metrics and comparison with the same baseline puzzles.
- `predictions/{image_id}.png`: all reconstructed images.
- `previews/01.png` through `03.png`: first three puzzles in index order, showing
  ground truth, shuffled input, and prediction. Selection does not depend on quality.
- `failure_analysis.json`: failed puzzle IDs, five lowest-position results,
  and known limitations.

Source/output overlap and an output directory containing the checkpoint are
rejected. Existing outputs with different configurations are rejected. Identical
configurations can be rerun. Checkpoint provenance, unchanged training/validation
metadata, and evaluation/training image separation are checked. Baseline comparison
requires the same index and selected metadata. Original images and ground-truth
positions never enter the scoring network or the assembly algorithm. Ground truth
is used for integrity verification, metrics, and labeled previews only.

## Actual results (October 19, 2026)

Checkpoint: `siamese_cpu_v1/best.pt`, epoch 2 from Phase 6. CPU inference, two
PyTorch threads. Each split contains 75 puzzles, 1,875 pieces, 3,000 assembled
borders, and 6,000 true-neighbor ranking queries.

| Metric | Validation CNN | Validation classical | Test CNN | Test classical |
| --- | ---: | ---: | ---: | ---: |
| Neighbor prediction accuracy | 29.77% | 95.85% | 28.03% | 97.70% |
| Fragment position accuracy | 5.23% | 83.36% | 5.01% | 88.32% |
| Assembly neighbor accuracy | 18.13% | 93.33% | 17.27% | 96.83% |
| Exact puzzles | 0/75 | 59/75 | 0/75 | 63/75 |
| Exact reconstruction rate | 0% | 78.67% | 0% | 84% |

Metric definitions:

- **Neighbor prediction accuracy:** for every true directional neighbor query,
  rank all 24 other pieces; count whether the top choice is the true neighbor.
  This measures local scores before assembly and matches Phase 6.
- **Fragment position accuracy:** pieces placed in their exact original row and
  column divided by all pieces. Global shifts are counted as errors.
- **Assembly neighbor accuracy:** correct directed RIGHT/DOWN relationships in
  the final arrangement divided by its 40 borders per puzzle.
- **Exact reconstruction rate:** puzzles with all 25 pieces in their original
  positions divided by all puzzles.

Mean CNN border cost is 0.6790017355 on validation and 0.6784785069 on test.
It is an optimization objective, not a calibrated reconstruction confidence.
Its units differ from classical RGB edge differences; do not compare these costs.

The first validation preview (bed and pillows) groups some similar colors/textures
while misplacing many fragments. This is consistent with the poor local neighbor
ranking measured in Phase 6, compounded by irreversible greedy decisions.
The evaluation does not establish which model or solver change would help most;
limited CPU training and coarse features remain hypotheses rather than measured
causes. No test-driven tuning or additional optimization was introduced here.

## Verification and limitations

`tests/test_reconstruction.py` adds nine tests, including parameterized invalid
inputs. Controlled directional scores recover a shuffled grid exactly; uniform
scores test tie handling and self exclusion. Image-only inference is repeatable,
uses all pieces, renders 320 x 320, and preserves weights. A synthetic checkpoint
and generated puzzles exercise validation/test evaluation, metric counts, saved
images, hash-identical reruns, overlap rejection, changed configuration rejection,
and training-image leakage rejection. Controlled scores verify solver logic;
they are not claims about learned-model accuracy.

The full suite passed: **74 tests in 56.21 seconds**. The targeted new tests passed:
**9 in 12.83 seconds**. `pip check` found no broken requirements; `git diff --check`
passed. Both real split runs completed with all 150 predictions saved.
Rerunning both splits reproduced all 164 output files with identical SHA-256
hashes, including summaries, arrangements, previews, and prediction images.

Supported inputs remain a complete 5 x 5 puzzle of upright, equally sized square
pieces. Missing pieces, arbitrary piece counts, rotations, irregular physical
jigsaw shapes, upload UI, and guaranteed reconstruction are not implemented.
No new dependency was added. Data, checkpoints, and generated outputs remain
ignored by Git. Phase 8 will separately address broader end-to-end evaluation
and failure analysis after approval.

Additional model improvements were separately approved after Phase 7. See
[CNN experiments](cnn_experiments.md) for the boundary model and subsequent
validation results. The Phase 7 figures above describe the original whole-piece
checkpoint and remain unchanged for comparison.
