# Phase 6: training and held-out pair evaluation

This phase trains the existing `siamese_v1` model. Neural puzzle assembly remains
Phase 7. Matplotlib is added for standalone learning curves.

## Commands

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.train --output artifacts/training/pilot --epochs 2 --train-puzzles 8 --validation-puzzles 2 --max-seconds 120
.\.venv\Scripts\python.exe -m jigsaw_ai.train
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_model --split validation
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_model --split test
```

Default full training uses 350 training puzzles (56,000 balanced pairs) and 75
validation puzzles (12,000 balanced pairs). No test pairs enter training or
checkpoint selection. The limited pilot is a plumbing/runtime check, not a final
performance experiment. Run `--help` for source, output, and setting overrides.

Default settings: seed 42, Adam learning rate 0.001, batch size 64, direction
weight 1.0, six maximum epochs, patience two, two CPU threads, and a 600-second
learning budget. Inputs remain RGB float32 / 255; no pretrained weights or
augmentation are introduced.

## Training and selection

Training uses backward propagation and Adam. Validation uses evaluation and
inference modes and cannot update weights. Binary loss averages over all pairs;
direction loss and direction accuracy use positive pairs only. Epoch loss
aggregation weights each batch's binary loss by example count and direction loss
by positive count, correctly handling negative-only and short batches.

Initial validation records random-initialization reference losses. Among trained
checkpoints, minimum validation BCE + weighted direction CE selects `best.pt`.
Stopping occurs at the epoch limit, patience limit, or learning-time budget.
Test results never choose settings or checkpoints.

The budget begins after dataset preparation and initial validation. It is checked
between training batches, allowing at least one batch. Final validation and
checkpoint reload verification still finish afterward, so it is not a strict
process wall-clock deadline. A truncated epoch is explicitly marked incomplete;
history records its actual examples and optimizer steps.

## Artifacts and reproducibility

The default Git-ignored output is `artifacts/training/siamese_cpu_v1/`:

- `config.json`: all settings, selected original IDs, source hashes, versions,
  architecture, normalization, and optimizer.
- `initial_validation.json`: untrained reference on balanced validation pairs.
- `history.json`: epoch losses, accuracies, counts, durations, completion flags.
- `best.pt` / `last.pt`: best-validation and latest checkpoints.
- `summary.json`: actual updates, stopping reason, best epoch, reload checks.
- `learning_curves.png`: actual loss and positive-direction accuracy curves.

Python, NumPy, PyTorch, and DataLoader seeds are fixed; deterministic PyTorch
algorithms and a fixed CPU thread count are used. Exact reproducibility across
different library versions and hardware is not guaranteed.

Checkpoints store versioned model weights, optimizer state, epoch, configuration,
validation metrics, label mapping, and PyTorch RNG state. Loading uses
`weights_only=True`, CPU mapping, and strict schema/architecture/label validation.
The trainer reloads the best checkpoint and checks that validation metrics match.

New training runs require empty output directories to preserve existing weights.
This CLI does not resume interrupted runs; completed checkpoints can still be
evaluated. Exact training resume would additionally require DataLoader shuffle
state and progress. Durations and serialized artifact bytes are not promised to
match across runs, even when weight computations are deterministic.

## Held-out evaluation

Default evaluation uses the best checkpoint and outputs to
`artifacts/model_evaluation/<split>/`. Options override checkpoint, source,
output, calibration, and baseline report paths.

Evaluate all 600 ordered non-self pairs per puzzle: 45,000 pairs per held-out
split, consisting of 6,000 positives and 39,000 negatives. Sigmoid compatibility
scores classify adjacency in any direction. Maximum validation F1 chooses the
threshold; test requires calibration tied to the exact checkpoint and dataset.

Neighbor ranking uses `P(compatible) * P(direction | compatible)`. For every one
of 80 true directed relationships per puzzle, rank all other 24 pieces. Original
positions produce evaluation labels only, never model inputs. Ties choose the
lowest shuffled index; self-candidates are excluded.

The encoder runs once per piece, caching 25 embeddings for all 600 pairs. Tests
compare cached and direct model forward scores. This speeds inference without
changing the architecture.

Evaluation saves config, summary, per-puzzle neighbor metrics, and low-performing
puzzle records. It also saves the first eight incorrect neighbor queries and
a `failure_examples.png` comparison of the queried piece, predicted neighbor,
and true neighbor when errors exist. These columns compare pieces; they are not
their physical arrangement for UP/DOWN directions. Validation also saves calibration. Reports contain binary
precision, recall, F1, accuracy, confusion counts, positive-pair direction accuracy,
and directional neighbor ranking accuracy. Scores are not calibrated guarantees.
No neural position accuracy or exact reconstruction rate is reported before
Phase 7 assembly.

Classical comparisons require identical held-out puzzle hashes. Both methods
classify all 600 pairs and rank all 24 candidates for the same 80 neighbor queries.
Each method's threshold comes only from validation. Classical exact reconstruction
accuracy is not compared to a CNN pair metric because they measure different tasks.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
```

Tests cover a tiny synthetic learning task, weighted losses, unchanged validation
weights, checkpoint reload, cached/direct inference equivalence, self exclusion,
split separation, calibration provenance, output protection, invalid settings,
and truthful handling of time-limited partial epochs.

Reference: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

## Actual experiment (October 18, 2026)

The pilot completed two epochs on eight training puzzles and two validation
puzzles: 40 optimizer updates, approximately 36.1 reported training seconds,
weights changed, and checkpoint reload verified. Validation direction accuracy
remained 25%; the pilot was not a successful generalization experiment.

A timing check of 20 in-memory training batches took about 10.44 seconds with
two threads and 8.44 seconds with four. The full experiment therefore used:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.train --epochs 3 --threads 4 --max-seconds 1200
```

All other settings remained at their documented defaults. It completed **two
full epochs**, 112,000 pair presentations, and **1,750 optimizer updates** on all
350 training originals. The third epoch was not started because the learning
budget was reached after epoch-two validation. Both recorded epochs are complete.
Reported training time including checkpoint reload verification was about
1,273.3 seconds (21.2 minutes), excluding data preparation, initial validation,
and plotting. The extra time beyond the budget completed validation/reload checks.

| Epoch | Train loss | Validation loss | Validation positive-direction accuracy |
| --- | --- | --- | --- |
| Initial model | — | 2.08066 | 25.00% |
| 1 | 1.88135 | 1.53021 | 59.83% |
| 2 | 1.35069 | 1.21654 | 74.38% |

The selected checkpoint is epoch 2. Reload reproduced its balanced-validation
metrics exactly. Training loss averages predictions from changing weights during
the epoch; validation evaluates the final weights. Validation loss being lower
than that training average is not by itself evidence of leakage.

### Frozen held-out results

Each split below contains 75 puzzles, 45,000 ordered pair examples, and 6,000
neighbor queries. The CNN compatibility probability threshold **0.5415676832**
was selected on validation and reused unchanged on test.

| Metric | CNN validation | CNN test | Classical test |
| --- | --- | --- | --- |
| Pair precision | 22.73% | 22.21% | 51.11% |
| Pair recall | 63.97% | 62.67% | 77.63% |
| Pair F1 | 33.54% | 32.80% | 61.64% |
| Pair accuracy | 66.21% | 65.76% | 87.12% |
| Direction accuracy on true pairs | 74.38% | 73.92% | Not measured in Phase 4 |
| Neighbor ranking accuracy | 29.77% | 28.03% | 97.70% |

CNN test confusion counts: TP 3,760; FP 13,166; TN 25,834; FN 2,240. Direction
classification got 4,435 of 6,000 true-pair directions correct. Neighbor ranking
got 1,682 of 6,000 queries correct. These tasks are different: identifying a
direction after being given a true pair does not establish ability to find that
neighbor among 24 candidates.

An always-negative classifier would score 86.67% pair accuracy on these examples
but zero pair F1. The chosen threshold maximizes validation F1, not accuracy.
Precision, recall, and neighbor ranking expose the model's current limitations.

The classical validation pair F1 and neighbor accuracy were 62.28% and 95.85%.
Both comparisons use the identical original-image splits and metadata hashes.
No neural position or exact-puzzle reconstruction metric has been calculated.
The classical solver's earlier 84% exact rate is a separate assembly result.

### Interpretation and verification

**This CNN underperforms the classical baseline.** The learning curves show
improvement during the short CPU run, but do not prove that longer training will
close the gap. Limited training, the small number of originals, and coarse
spatial features are possible contributors; no ablation establishes a cause.
Do not change settings based on these test results and then treat the same test
set as a fresh independent benchmark.

The deterministic first-error sheets were inspected. In validation puzzle
`7c3b876233adde13`, the model selects similarly colored bedding regions instead
of the true boundary continuation. Test puzzle `18c620982f888c5d` shows incorrect
choices among gray land/water regions. These are illustrative cases, not a count
of all failure causes. Per-puzzle results cover every held-out puzzle.

- Full automated suite: **65 passed, 0 failed, no warnings**.
- Additional targeted training/evaluation suite: 12 passed.
- Tiny synthetic learning, early stopping, exact seeded weights, validation-only
  selection, unchanged validation weights, checkpoint reload, mask aggregation,
  cached/direct inference, self exclusion, time budgets, source overlap, and
  calibration provenance checks passed.
- `pip check`: no broken requirements.
- Checkpoint and evaluation outputs are ignored by Git; no remote is configured.
- No GPU training was attempted; this is a CPU float32 experiment.
- Full-run training artifacts occupy 4,830,664 bytes; held-out evaluation artifacts
  occupy 440,079 bytes. Pilot artifacts remain in a separate directory.

The best/last checkpoints, configs, metrics, learning curves, and failure examples
are saved locally. Model and threshold stayed fixed during test evaluation.
Nothing was committed or pushed.
