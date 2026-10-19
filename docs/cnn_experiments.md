# Approved CNN improvement experiments

These experiments extend the model work in Phases 5–7; they do not start Phase 8
or any backend/UI phase. The old whole-piece CNN and its reports remain intact.

## What changed and why

The old CNN ranked the true directional neighbor first on only 29.77% of the
6,000 validation queries. Its reconstructed validation images placed 5.23% of
pieces correctly and solved none of 75 puzzles exactly. The same greedy solver
with classical edge scores solved 59/75. These results motivated improving
local scores before introducing a more complicated global solver.

The new `BoundaryCNN` is a small Siamese network. It extracts four RGB strips
from each fragment, each 64 pixels along the edge and four pixels deep. Strips
are oriented with the touching edge at depth zero. The same convolutional encoder
processes all pieces and all sides. It uses a 3-by-1 convolution (3 RGB channels
to 8 features), ReLU, an 8-to-8 pointwise convolution, and ReLU. Comparing a side
of A with the opposite side of B gives absolute feature differences; average
along the edge and combine with learned nonnegative channel/depth weights.
A learned scale and bias turn the resulting distance into a matching logit.

Unlike the original whole-piece architecture, this network directly preserves
edge detail. Its opposite-direction scores agree by construction: A-RIGHT-B has
the same cost as B-LEFT-A. It is a learned boundary metric, not a general semantic
image encoder and not the original two-head architecture with more training.

Initialization deliberately includes an **RGB-continuity prior**: the first three
encoder features start as RGB identity channels, and depth-zero comparisons
receive most weight. Other features start with small seeded random weights.
The initial validation score is saved before any training. This is essential:
an architectural prior is useful, but its benefit must not be attributed to
training. The pure CNN scorer does not mix in a separate classical score at
inference; the hybrid variants below do so explicitly.

## Experiment design

All runs use the same 350 training originals and 75 validation originals, seed 42,
Adam with learning rate 0.001, CPU execution with two PyTorch threads, and
deterministic operations. One puzzle is processed per optimizer step. All 80
true directed neighbor queries per puzzle supply supervision. No test puzzles
are loaded, scored, or used during training and validation selection. A separate
confirmation command evaluates only the already frozen choices afterward.

Three training runs start from identical model weights:

1. **Random negatives, 4 epochs:** balanced binary cross entropy using each true
   directional neighbor and one incorrect candidate sampled uniformly among
   the other 23 eligible pieces. Self pairs are excluded.
2. **Hard negatives, 4 epochs:** the same binary objective, sampling the negative
   from the model's eight highest-scoring incorrect candidates. The true neighbor
   and self pair are excluded; candidates are refreshed as the network changes.
3. **Neighbor ranking, 12 epochs:** cross entropy ranks the true neighbor against
   all 24 nonself candidates, rather than making independent binary decisions.
   Compare the best checkpoint through epoch 4 with the best through epoch 12
   to measure whether longer training helps this objective.

Boundary queries which point outside the original image have no neighbor and
are excluded from training targets. A piece that is a neighbor in a different
direction can still be a negative for the queried direction. There is no
independent relative-direction classification head in this architecture.

Validation neighbor accuracy selects checkpoints, with validation ranking loss
breaking ties. The untrained initial model remains eligible, preventing a
training regression from replacing a better initialization. Each run saves its
initial score, every complete epoch, selected model and optimizer states,
configuration, RNG states, and reload-verified validation metrics. Exact training
resume is not implemented. Comparing training loss values across the binary
and ranking objectives is not meaningful; validation ranking loss is comparable.

After selecting the CNN by validation neighbor accuracy, ten assembly settings
are evaluated: CNN weights 1, 0.75, 0.5, 0.25, and 0, each with and without swap
refinement. CNN and classical costs are individually divided by their mean
over all nonself pairs and directions within each puzzle before interpolation.
Weight 1 is pure CNN; weight 0 is pure classical; intermediate weights are
hybrids. This normalization uses input pixels and predictions, not ground truth.

All settings use the existing all-25-anchor greedy assembler. Optional refinement
evaluates all 300 possible swaps of two placed pieces, accepts the best strictly
improving swap, and repeats for at most 20 rounds. It preserves the permutation
and cannot increase the chosen objective; it does not guarantee accuracy gains
or a global optimum. Overall assembly selection maximizes exact puzzles, then
position accuracy, then correct assembled borders on validation. A separate
best pure-CNN configuration is saved even if a hybrid/classical setting wins.

## Commands and artifacts

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.improve_cnn
.\.venv\Scripts\python.exe -m jigsaw_ai.plot_cnn_experiments --gallery
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_boundary
```

Defaults write to `artifacts/cnn_experiments/boundary_v1`, which must be new/empty
when starting experiments. Generated artifacts and checkpoints remain ignored
by Git. Training arguments allow a different source/output and epoch counts.
Use a new output directory for a new experiment; no old checkpoint is overwritten.

Each objective has `initial_validation.json`, `history.json`, `summary.json`,
and `best.pt`. The experiment root has input provenance, training comparison,
all ten per-puzzle assembly reports, summary, a longer-training comparison,
learning curves, and a reconstruction gallery using the first three validation
images. Predictions for every validation puzzle are saved separately under
`pure_cnn/` and `selected/`. Best pure-CNN predictions can include refinement;
their exact configuration is identifiable in `assembly_comparison.json`.

For independent inference, supply 25 RGB 64-by-64 PNG fragments. The solver reads
no puzzle metadata, positions, or reference image:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.solve_boundary --pieces PATH_TO_PIECES --checkpoint PATH_TO_BEST_PT --output artifacts/my_boundary_result
```

By default this uses the CNN alone and greedy assembly. `--refine` enables swaps;
`--cnn-weight 0.75` explicitly enables a hybrid. The output directory must be
new/empty. Outputs are `reconstruction.png` and `arrangement.json`, which records
filenames in row-major order, input/checkpoint hashes, method settings, and cost.
It does not invent accuracy when ground truth is unavailable. The model format
is `boundary_cnn_v1`; use this command rather than the original two-head model's
evaluation command to load these checkpoints.

## Actual results (October 19, 2026)

All three runs completed: 4 random-negative epochs (1,400 updates), 4 hard-negative
epochs (1,400 updates), and 12 ranking epochs (4,200 updates). All selected weights
changed from initialization and all checkpoint reloads reproduced validation
metrics exactly. The encoder and metric have **186 trainable parameters**.
Reported training times, including validation, were 131.51, 134.82, and 453.59
seconds respectively; loading/caching and later assembly experiments are extra.

Validation training comparison (6,000 neighbor queries):

| Model/checkpoint | Best epoch | Correct neighbors | Neighbor accuracy |
| --- | ---: | ---: | ---: |
| Original whole-piece CNN | 2 | 1,786 | 29.77% |
| Classical edge baseline | No training | 5,751 | 95.85% |
| Boundary initialization with RGB prior | 0 | 5,750 | 95.83% |
| Boundary CNN, random negatives | 4 | 5,769 | 96.15% |
| Boundary CNN, hard negatives | 4 | 5,770 | 96.17% |
| Boundary CNN, ranking | 12 | 5,795 | 96.58% |

Ranking training initially reduced accuracy: epoch 4 reached 95.43%, below the
initial 95.83%. Through epoch 4, the initial checkpoint remained the best eligible
choice. Continuing to epoch 12 increased that best score by 45 correct neighbors,
or 0.75 percentage points. The longer run helped in this experiment; that is not
a guarantee for other architectures or datasets. Hard-negative training beat
random-negative training by only one validation query, which is too small to
claim a reliable general advantage from a single seed.

Full validation assembly comparison:

| Model/scoring | Assembly | Position accuracy | Correct assembled borders | Exact puzzles |
| --- | --- | ---: | ---: | ---: |
| Original whole-piece CNN | Greedy | 5.23% | 18.13% | 0/75 |
| Classical | Greedy | 83.36% | 93.33% | 59/75 |
| Classical | Greedy + swaps | 83.47% | 94.40% | 59/75 |
| Initial boundary model | Greedy + swaps | 83.31% | 94.37% | 59/75 |
| Random-negative CNN | Greedy + swaps | 87.73% | 96.13% | 62/75 |
| Hard-negative CNN | Greedy + swaps | 87.68% | 96.03% | 62/75 |
| Ranking CNN alone | Greedy | 89.12% | 95.57% | 63/75 |
| Ranking CNN alone | Greedy + swaps | 89.71% | 96.53% | 64/75 |
| 75% ranking CNN / 25% classical | Greedy + swaps | 90.24% | 96.73% | 64/75 |

The initial/each-trained-checkpoint comparison is saved separately as
`checkpoint_assembly_comparison.json` and per-puzzle files under
`checkpoint_ablation/`. It makes the large architectural/initialization gain
visible separately from the smaller but measured training gain. The ten scoring
and refinement settings are in `assembly_comparison.json`; neither hybrids nor
refinement consistently improved every setting.

**Frozen test confirmation:** after all validation choices above were saved,
`evaluate_boundary` evaluated exactly two previously specified configurations:
the best pure-CNN assembly and the overall validation-selected hybrid. Each used
the same 75 existing test puzzles (1,875 pieces, 3,000 borders, 6,000 neighbor
queries). No model/settings were changed using these results.

| Method | Neighbor accuracy | Position accuracy | Correct assembled borders | Exact puzzles |
| --- | ---: | ---: | ---: | ---: |
| Original whole-piece CNN, greedy | 28.03% | 5.01% | 17.27% | 0/75 (0%) |
| Classical baseline, greedy | 97.70% | 88.32% | 96.83% | 63/75 (84%) |
| Pure boundary CNN, greedy + swaps | 98.30% | 91.89% | 97.97% | 68/75 (90.67%) |
| Validation-selected 75% CNN hybrid, greedy + swaps | 98.25% | 90.40% | 97.80% | 67/75 (89.33%) |

The hybrid's validation advantage did not carry over to test: pure CNN did better
there. Both configurations were frozen before test, and the overall selection
remains the validation-selected hybrid; the test result was not used to retune
the weight or swap policy. The local-neighbor score improvement and the
validation greedy-only comparison help separate scoring gains from swap gains.
These results are confirmation on a previously inspected test set, not a fresh
independent final evaluation and not evidence of a universal win over classical
matching. Seven test puzzles still fail exact reconstruction with pure CNN.

Selected trained checkpoint:
`artifacts/cnn_experiments/boundary_v1/ranking/best.pt` (epoch 12).
Its SHA-256 is
`2537358f0d51b6a3b25a65030b279a586601fdbead3570c837ba8e6baa0e0d79`.
The frozen audit records the checkpoint hash, selection-report hash, original
validation choices, and test metadata hashes. Artifacts are under
`artifacts/cnn_experiments/test_audit_v1`, with all 75 pure and 75 hybrid prediction
images, per-puzzle counts, and aggregate metrics.

Use the trained CNN alone, with the validation-selected pure-CNN swap policy:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.solve_boundary --pieces data/puzzles/validation/7c3b876233adde13/pieces --checkpoint artifacts/cnn_experiments/boundary_v1/ranking/best.pt --output artifacts/my_ranked_cnn_result --refine
```

For the overall validation-selected hybrid, additionally pass `--cnn-weight 0.75`.
No resizing, random augmentation, pretrained weights, GPU, or extra image downloads
were introduced in these experiments.

## Tests and checks

The final `python -m pytest -q` passed **83 tests in 45.28 seconds**. The new
targeted suite also passed **9 tests in 16.58 seconds**. No failed tests or broken
requirements were reported; `git diff
--check` passed.

The new tests cover strip orientation against actual touching-edge MAE, all
three losses and finite encoder gradients, reciprocal scores, swap objective
monotonicity and piece uniqueness, trained checkpoint reload, inference without
ground truth, repeatable output, overwrite and format protection, and the
complete tiny experiment pipeline. Test images are deliberately withheld until
after selection in the integration test; only the frozen audit needs them.
A checkpoint changed after selection is rejected. A real standalone CLI run
also reconstructed the first validation puzzle using the trained hard-negative
checkpoint and saved all 25 unique filenames.

## How to interpret results

The subsequent [search improvement](search_improvements.md) keeps this checkpoint
frozen and reports a separate 50-image fresh holdout. Its cohort and figures
must not be mixed with the original 75-image test comparison above.

Compare all new results with the **validation** figures from Phases 6–7, not
their test figures. The same 75 originals are used. Local neighbor ranking,
absolute piece position, correct assembled borders, and exact puzzles are
different metrics; definitions remain those in `reconstruction.md`.

These are validation-selected experiments, not a new independent final test.
The previous test set has already been inspected. A fresh held-out image set
will be needed to substantiate final generalization after model development.
This suite does not claim statistical significance from one seed or assume
that harder negatives, longer training, blending, or refinement always help.
It does not yet handle missing pieces, rotations, arbitrary grids, irregular
physical jigsaws, or damaged boundaries. No new dependency was added.
