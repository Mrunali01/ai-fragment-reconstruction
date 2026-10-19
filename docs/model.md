# Phase 5: untrained Siamese CNN

This phase implements and checks computations only. There is no optimizer,
training loop, learned checkpoint, accuracy measurement, or neural reconstruction.
The classical baseline remains the only evaluated solver.

## Architecture

Two fragment tensors enter the **same encoder instance**, which shares all its
weights between branches. No pretrained weights are downloaded.

| Per-branch stage | Output shape (N = batch size) |
| --- | --- |
| RGB input in [0,1] | N x 3 x 64 x 64 |
| 3x3 convolution, 16 channels, ReLU, 2x2 max pool | N x 16 x 32 x 32 |
| 3x3 convolution, 32 channels, ReLU, 2x2 max pool | N x 32 x 16 x 16 |
| 3x3 convolution, 64 channels, ReLU, 2x2 max pool | N x 64 x 8 x 8 |
| Adaptive average pool to 4x4; flatten | N x 1024 |
| Linear projection and ReLU | N x 128 |

Convolutions use padding 1 and stride 1. Retaining 4x4 spatial cells preserves
coarse location cues instead of reducing every fragment to a single spatial
average. This small architecture is an initial hypothesis, not an established
improvement over edge matching.

Concatenate `[embedding_first, embedding_second]` to form N x 256 ordered features.
A shared pair head applies Linear(256,128) and ReLU. Two final linear heads output:

- `compatibility_logits`: N raw values for the binary compatibility task.
- `direction_logits`: N x 4 raw values in LEFT, RIGHT, UP, DOWN order.

The encoder is shared; concatenation preserves which fragment came first. Using
only absolute differences would erase this order and make relative direction
ambiguous. The architecture does not enforce exact swap symmetry for compatibility
or exact opposite-direction behavior; those properties would need to be learned
or explicitly constrained later. No direction prediction is meaningful for a
negative pair.

No batch normalization or dropout is used. Single-example batches work without
special cases. The model requires matching nonempty N x 3 x 64 x 64 inputs with
finite values and the same floating dtype/device as its parameters.

## Tensor preparation

`image_tensor()` copies RGB Pillow pixels into an owned NumPy float32 array, then
converts it to a contiguous tensor in channel-first order, dividing by 255. It
does not stretch, crop, rotate, or augment the fragments. NumPy also provides
PyTorch's tensor interoperability support; torchvision is not needed here.

`collate_pairs()` stacks existing Phase 3 samples and produces:

| Key | Shape / dtype |
| --- | --- |
| first / second | N x 3 x 64 x 64, float32 |
| compatible | N, float32 containing 0 or 1 |
| direction | N, int64; 0..3 for positives, -1 for negatives |

It validates labels and replaces Phase 3's negative `None` with -1. This custom
collator works with `torch.utils.data.DataLoader`; the existing `PairDataset`
does not need a second wrapper class. Ground-truth coordinates, shuffle seeds,
and prepared target images are excluded from model inputs.

```python
from pathlib import Path
from torch.utils.data import DataLoader
from jigsaw_ai.pairs import PairDataset
from jigsaw_ai.tensor_data import collate_pairs
from jigsaw_ai.siamese import SiameseCNN
from jigsaw_ai.model_loss import pair_loss

dataset = PairDataset(Path("data/puzzles"), "train")
batch = next(iter(DataLoader(dataset, batch_size=8, collate_fn=collate_pairs)))
model = SiameseCNN()
outputs = model(batch["first"], batch["second"])
losses = pair_loss(outputs, batch["compatible"], batch["direction"])
losses["loss"].backward()  # Verify gradients; this does not update weights.
```

## Loss definition

`total = compatibility_BCE + direction_weight * positive_direction_CE`

The default direction weight is 1.0. Binary cross entropy with logits averages
over every pair. Cross entropy averages over positive pairs only. These losses
consume raw logits; do not apply sigmoid or softmax before calling them.

For a batch with no positives, direction loss is a differentiable zero. Negative
direction entries do not contribute gradients. The function validates target
shapes, devices, dtypes, binary labels, direction labels, and finite inputs.
Labels should move to the same device as model outputs in future training code.

Sigmoid and softmax may later turn logits into scores for inference. Those scores
will still require evaluation and should not be called reliable confidence just
because they lie between zero and one.

## Installation and checks

For this CPU Windows environment:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m jigsaw_ai.inspect_model
.\.venv\Scripts\python.exe -m pip check
```

The dependency configuration records the supported torch range; it does not
force a CPU build for every platform. Use the official installation selector for
other platforms: https://pytorch.org/get-started/locally/.

`inspect_model` selects four positive and four negative real **training** pairs,
sets a reproducible initialization seed, and checks forward/backward computations.
The CLI uses two CPU threads to keep this small computation check economical.
It separately checks a single example and a negative-only batch. It verifies
weights remain unchanged, and takes no optimizer steps. The report is written to
Git-ignored `artifacts/phase5_model_check.json`. `--source`, `--output`, and
`--seed` override the defaults.

Report losses describe random initial computation, not trained performance.
The script reads no validation/test pairs. No trained checkpoint is saved.

References:

- Binary loss: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- Direction loss: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

Training configuration, optimization, validation monitoring, checkpoints, and
unseen-image evaluation are implemented separately in approved Phase 6; see
[training documentation](training.md). The results below record the untrained
Phase 5 computation check, not the trained checkpoint's performance.

## Actual verification results (October 17, 2026)

Environment: Python 3.13.2, PyTorch 2.14.1+cpu, NumPy 2.5.2, and Pillow 12.3.0.
The installed build reports CUDA unavailable; GPU execution was not tested.

| Check | Actual result |
| --- | --- |
| Trainable parameters | 188,325 |
| Real batch | 4 positive + 4 negative training pairs |
| Input shape per branch | [8, 3, 64, 64] |
| Compatibility logits | [8] |
| Direction logits | [8, 4] |
| Single-example outputs | [1] and [1, 4] |
| Parameter gradients | All present and finite |
| Gradients through both input branches | Finite and nonzero |
| Negative-only direction loss | Exactly 0 |
| Weights after backward | Unchanged |
| Optimizer steps | 0 |
| Full automated suite | 53 passed, 0 failed, no warnings |

The real-batch initial total loss was 2.0369491577148438 (compatibility
0.6933774352073669; direction 1.3435717821121216). These are diagnostic values
from seeded random weights, not trained accuracy, quality, or improvement over
the baseline. Real inputs for this check came from training puzzle
`f1bafb99791248a2`; it is not a generalization experiment.

The first 52-test run passed with a missing-NumPy bridge warning. NumPy was then
added to tensor preparation and dependencies, removing the warning. Two NumPy
2.5.4 download attempts ended in remote connection resets. Installation succeeded
with NumPy 2.5.2 selected through the official PyTorch package index:

```powershell
.\.venv\Scripts\python.exe -m pip install "numpy>=2.1,<3" --index-url https://download.pytorch.org/whl/cpu --timeout 30 --retries 2
```

A subsequent test also checks that negative-only direction loss remains zero
even with very large finite logits; the zero construction avoids an overflowing
sum. Tests additionally verify RGB channel order and scaling, custom DataLoader
collation, shared module use, ordered pair fusion, batch sizes one and four,
positive-only and negative-only backward passes, masked gradients, hand-computed
losses, invalid inputs, and an in-memory untrained state-dictionary round trip.

`pip check` found no broken requirements. Repeating the real-data check with the
same seed reproduced the report byte for byte (SHA256 unchanged). Git checks
confirmed the report is ignored and no remote is configured. No commit or push
was performed.
