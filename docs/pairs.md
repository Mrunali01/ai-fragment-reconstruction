# Phase 3: fragment pair data

The pair dataset teaches a future model whether two pieces touch and, if so,
which direction the second piece occupies relative to the first. No model is
trained in this phase. No new dependencies are installed.

## Label rules

For positions `(row_a, column_a)` and `(row_b, column_b)`, compute
`dr = row_b - row_a` and `dc = column_b - column_a`.

| Relationship | Offset `(dr, dc)` | Direction ID |
| --- | --- | --- |
| LEFT | (0, -1) | 0 |
| RIGHT | (0, 1) | 1 |
| UP | (-1, 0) | 2 |
| DOWN | (1, 0) | 3 |

Compatibility is 1 when `abs(dr) + abs(dc) == 1`; diagonals do not count.
Self-pairs are excluded. Non-neighbors receive compatibility 0 and direction
`null` in exported JSON. The loader returns Python `None` for their direction.
Future training must mask negatives out of the direction loss. Do not assign
negative pairs a fake direction or pass `None` directly to a tensor collator.

## Counts and sampling

Each 5x5 puzzle has 20 horizontal and 20 vertical borders. Including both orders
gives 80 positive pairs: 20 for each direction. There are 25 x 24 = 600 possible
ordered non-self pairs, leaving 520 valid negative candidates.

Keep all 80 positives and sample 80 unique ordered negatives without replacement.
Reverse negative pairs may both appear; they are distinct ordered examples.
Shuffle the combined 160 records using a local RNG seeded from the master seed
and puzzle ID. Seeds do not change the positive set. Negatives and order are
fixed across epochs for now, simplifying evaluation and reproducibility.

Balance is artificial: the full candidate space has more negatives. Later model
evaluation should also consider the full candidate space and ranking metrics;
balanced pair accuracy alone would not establish reconstruction quality. Hard
negative mining and direction-conditioned compatibility are future extensions,
not part of this initial pair definition.

## Loader

```python
from pathlib import Path
from jigsaw_ai.pairs import PairDataset

dataset = PairDataset(Path("data/puzzles"), split="train", seed=42)
sample = dataset[0]
# sample["first"], sample["second"]: independent RGB Pillow images, 64x64
# sample["compatible"]: 0 or 1
# sample["direction"]: integer 0..3 for positives, None for negatives
```

`PairDataset` is a Python sequence, not yet a PyTorch Dataset. It stores lightweight
pair records in memory and loads only the two requested fragment images. Its
constructor verifies each selected puzzle's saved-file integrity. Each access
also checks the requested image hashes. Future tensor conversion belongs to the
model/training phases.

Only fragment pixels are model inputs. Original positions, image targets, and
shuffle seeds are used for labels or debugging and are not returned as inputs.
`dataset.records` separately exposes IDs, relative paths, hashes, and labels.
All pairs are within one puzzle. Split assignment follows the existing index;
no pair or original image is moved between training, validation, and test.
Duplicate puzzle IDs across splits and mismatched index metadata are rejected.

## Commands and outputs

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.generate_pairs
.\.venv\Scripts\python.exe -m jigsaw_ai.audit_pairs
.\.venv\Scripts\python.exe -m pytest
```

Generation accepts `--source`, `--output`, and `--seed`. Defaults are
`data/puzzles`, `data/pairs`, and 42. The audit accepts `--source` and `--pairs`.

Git-ignored `data/pairs/` contains:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: one ordered pair per line,
  referencing existing fragment files relative to the puzzle root.
- `config.json`: seed, label mapping, ratio, Python version, and source hashes.
- `summary.json`: actual counts by split and positive direction.
- `audit.json`: independent validation counts.
- `*_examples.png`: four positive directions and four negative examples per split.
  Each row displays first and second pieces side by side, even for vertical labels;
  the preview layout is not their intended physical arrangement.

The independent audit reads exported records and original coordinates rather
than generating the pairs again. It rejects incorrect labels, directions,
self-pairs, duplicate ordered examples, cross-split records, and mismatched image
references; it verifies per-puzzle counts and coverage of all true neighbors.
Image bytes were verified by the loader during generation; the label audit
compares recorded hashes but does not reread all image bytes.

Reruns with unchanged inputs overwrite only this generator's known output files.
Changed settings or source metadata require a new directory. Nonempty unrelated
directories and directories overlapping the puzzle source are rejected. A full
rerun repairs interrupted pair manifests using bounded atomic file replacement.
Run one generator per output directory. Do not manually edit generated labels.

## Actual counts

| Split | Puzzles | Positive | Negative | Total |
| --- | --- | --- | --- | --- |
| Training | 350 | 28,000 | 28,000 | 56,000 |
| Validation | 75 | 6,000 | 6,000 | 12,000 |
| Test | 75 | 6,000 | 6,000 | 12,000 |
| Total | 500 | 40,000 | 40,000 | 80,000 |

These are labeled examples, not 80,000 independent source images and not measured
model predictions. Generalization still depends on only 500 original images.

## Verification results (October 16, 2026)

- Full test suite: **35 passed, 0 failed**.
- Independent label audit: **all 80,000 exported pairs checked successfully**.
- Training direction counts: 7,000 positives for each of LEFT, RIGHT, UP, DOWN.
- Validation and test direction counts: 1,500 positives per direction per split.
- Full real-data export rerun: all output file SHA256 hashes unchanged.
- Three visual example sheets generated and inspected.
- Local pair output size: 35,191,069 bytes (about 35 MB); image files are referenced,
  not duplicated into the pair directory.
- `pip check`: no broken requirements. No dependencies were added.
- Git ignore checks confirmed pair outputs are excluded; no remote is configured.

Tests independently calculate adjacency from coordinates and check the reverse
direction of every positive pair. They also verify absence of self-pairs and
duplicate ordered records, deterministic sampling, changed-seed negative samples,
RNG isolation, actual loaded pixels, split isolation, corruption detection,
source/index disagreement, protected output settings, export tampering, and
byte-identical independent exports and reruns.

No model accuracy, reconstruction accuracy, or learned confidence is reported.
