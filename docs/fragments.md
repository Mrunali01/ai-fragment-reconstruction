# Phase 2: fragments and ground truth

## What this phase produces

Each Phase 1 manifest image becomes one **320 x 320 RGB puzzle**, cut into a
**5 x 5 grid of 64 x 64 pieces**. All 25 pieces are retained and shuffled without
individual rotations. These are rectangular image tiles, not physical jigsaw
pieces with tabs. No compatibility model, training pairs, or solver is implemented.

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.generate_puzzles
.\.venv\Scripts\python.exe -m pytest
```

Options: `--source` (default `data/open_images`), `--output` (default
`data/puzzles`), and `--seed` (default 42). Source and output directories must not
overlap. All generation is local; no additional dependency or download is needed.

## Preprocessing and orientation

1. Verify the source bytes against the Phase 1 manifest checksum.
2. For Open Images with a known `Rotation`, rotate the stored RGB pixels by that
   many degrees counterclockwise. Do **not** also apply EXIF rotation.
3. When source rotation is unknown, apply EXIF if present and record the unknown
   source orientation. Do not guess orientation from image content.
4. Resize proportionally with Lanczos so the shorter side becomes 320 pixels.
   Round the longer side to the nearest integer; reject inputs requiring upscaling.
5. Take a 320 x 320 center crop. If there is an odd extra pixel, floor the left/top
   offsets. Save sizes, crop coordinates, and orientation policy in metadata.

Open Images explains that its mirrored JPEGs had orientation metadata stripped
without rotating the pixels. Its CSV angle restores the intended orientation:
https://storage.googleapis.com/openimages/web/2018-05-17-rotation-information.html

For this collection, five images have nonzero angles and 58 have unknown source
rotation. The unknown group may still contain incorrectly oriented images.
The single nonzero-rotation preview is a visual example; all five angles are
applied by the same tested preprocessing function.

**The prepared crop is the reconstruction target**, not the uncropped original
photo. Center cropping removes edge content but preserves object proportions.
Downloaded source images remain untouched. PNG output avoids introducing JPEG
compression differences at touching piece edges.

## Shuffling and labels

`create_fragments()` builds all 25 `(row, column)` positions in row-major order,
shuffles them with a local seeded random generator, and cuts the corresponding
image regions. A per-image seed is derived from SHA256 of `master_seed:image_id`.
This avoids depending on processing order or altering Python's global RNG.

Saved piece IDs describe **shuffled order**, so `piece_000` does not reveal its
correct row or column. `puzzle.json` stores the ground truth separately:

```json
{
  "piece_id": "piece_000",
  "path": "pieces/piece_000.png",
  "shuffled_index": 0,
  "original_row": 2,
  "original_column": 4
}
```

This is an illustrative record; each actual record also includes its file SHA256.
Coordinates are zero-based: `(0, 0)` is the top-left and `(4, 4)` the bottom-right.
Future inference must not use original positions, seeds, or the prepared target
to score pieces. Those fields are for supervised labels and evaluation only.

## Output layout

```text
data/puzzles/
  config.json              Master seed, dimensions, source manifest hash, versions
  index.json               Puzzle metadata paths with preserved source splits
  summary.json             Verified generation counts
  train/<image_id>/
    prepared.png           Ground-truth target
    puzzle.json            Piece order, original positions, attribution, preprocessing
    pieces/piece_000.png   First shuffled piece
    ...
    pieces/piece_024.png   Last shuffled piece
  validation/<image_id>/   Same format
  test/<image_id>/         Same format
  previews/
    train.png
    validation.png
    test.png
    rotation_example.png
```

Previews show the prepared target, shuffled pieces, and an arrangement rebuilt
**using ground-truth labels**. This is a data-integrity demonstration, not a model
prediction or a measured solver reconstruction rate. Attribution, source links,
and modification descriptions are preserved in each puzzle's metadata.

## Verification and reruns

`verify_puzzle()` reopens every saved piece, checks file hashes, confirms 25 valid
pieces with a complete permutation of grid positions, and rebuilds the image.
Every RGB pixel must equal the saved prepared target. This proves no pixels were
lost, duplicated spatially, or changed by storage. Visually identical regions can
still occur naturally and will be a reconstruction challenge in later phases.

Rerunning with identical inputs verifies existing outputs and reuses them. A
changed master seed, source manifest, Python version, or Pillow version requires a
new output directory. These versions are recorded because preprocessing and
random-library behavior should not be assumed identical across environments.
Corrupted existing files stop the run; they are not silently replaced.

An interrupted new puzzle can leave a partially written puzzle directory. A
subsequent run reports missing or invalid files; use a fresh output directory or
explicitly repair that interrupted output. Automatic deletion is intentionally
not performed. Run one generator process per output directory.

All generated files are under Git-ignored `data/`. Source split assignments are
checked before generation and copied without moving images between splits.

## Tests

Tests cover pixel-exact synthetic round trips, all piece coordinates, dimensions,
local RNG isolation, repeatable and differing seeds, metadata rotation parsing,
counterclockwise rotation without double-applying EXIF, resize/crop geometry,
missing or duplicate positions, small/wrong-format images, saved-file tampering,
source split disagreement, output overlap, preservation of unrelated files, and
byte-identical independent generation and reruns.

## Verified results (October 15, 2026)

| Check | Actual result |
| --- | --- |
| Generated puzzles | 500 |
| Preserved splits | 350 training / 75 validation / 75 test |
| Saved fragment PNGs | 12,500 |
| Prepared target PNGs | 500 |
| Puzzle metadata files | 500 |
| Pixel-exact saved-file round trips | 500 passed |
| Prepared / piece dimensions | 320 x 320 / 64 x 64 |
| Nonzero source rotations applied | 5 |
| Unknown source orientation | 58 images |
| Generated directory size | 203,487,919 bytes (about 203 MB or 194 MiB) |
| Full automated test suite | 31 passed, 0 failed |
| Dependency consistency | `pip check`: no broken requirements |
| Full real-data rerun | Every generated file had the same SHA256 before and after |
| Preview review | All four panels visually inspected |

Executed `.\.venv\Scripts\python.exe -m pytest`,
`.\.venv\Scripts\python.exe -m jigsaw_ai.generate_puzzles`, and
`.\.venv\Scripts\python.exe -m pip check`. The rerun called `generate_dataset()`
over the same real collection while hashing every generated file before and
after. Git checks confirmed generated outputs are ignored and no remote exists.

No learning or automatic puzzle solving has occurred. A ground-truth round trip
checks generated data integrity; it must not be reported as 100% solver accuracy.
