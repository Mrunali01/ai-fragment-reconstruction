# Frozen CNN search improvement and a fresh holdout

This approved improvement revisits Phase 7 assembly. It does not train a new CNN
or begin backend/UI phases. The existing boundary CNN checkpoint and all earlier
results remain intact.

## What changed

The previous solver tries 25 greedy starts and then improving two-piece swaps.
That can get stuck when repairing a row requires moving several pieces together.
`group_search.py` adds generic moves over complete 5 x 5 arrangements:

- Swap any two pieces.
- Cyclically shift one row or column.
- Exchange complete rows or complete columns.
- Shift the complete grid cyclically.
- Exchange two disjoint 2 x 2 blocks.

These operations rearrange pieces without rotating their pixels. Every candidate
uses all 25 pieces exactly once. No image IDs, reference images, or original
positions enter candidate generation or selection.

Two fixed alternatives were compared on the existing 75 validation originals:
20 rounds of best improving group moves, and the same search followed by a
beam of four alternatives for four expansion steps. The returned arrangement
never has higher matching cost than the starting arrangement. A lower matching
cost does not guarantee more correct positions, which is why actual ground-truth
metrics and regressions are reported too.

## Overfitting precautions

The seven previously diagnosed test failures motivated the general move families,
so the old 75-image test set is considered development information here. It was
not used as a new final benchmark. This change does not memorize their IDs or
reference layouts.

Only two new search settings were considered, with fixed budgets. The existing
swap solver remained eligible. Validation exact reconstructions selected the
method, with correct positions and then borders as tie breakers; exact ties
retain the incumbent. The CNN weights and scoring objective stayed frozen.

After validation selected beam search, its selection-report hash and the CNN
checkpoint hash were recorded in an acquisition protocol **before downloading
or evaluating the fresh cohort**. The protocol fixed 50 images, a SHA-256
candidate ordering with seed 20261019, and all exclusion rules.

The fresh images came from previously unused candidates in the cached official
Open Images test metadata prefix. Exclusions covered all 500 original development
IDs, their known photographer-profile URLs, matching source URLs, file/pixel
checksums, and prepared-image 64-bit difference hashes within Hamming distance
four. The same near-duplicate check was applied within the new cohort. The actual
minimum hash distance was 12. This is a useful filter, not an exhaustive semantic
near-duplicate detector or proof against photographer aliases.

Per-image current CC BY 2.0 license evidence and attribution were checked using
the existing acquisition code. Image decode, minimum size, and puzzle round trips
were verified. Selection was based only on the frozen ordering, licensing, image
validity, and duplicate rules, never on model performance. Nineteen candidates
were rejected: 15 HTTP 404, two HTTP 403, one unsupported source URL, and one
changed license. The 50 accepted source images total 14,900,243 bytes (14.9 MB).
The bounded metadata pool and these exclusions limit population representativeness.

The fresh evaluation scored only the incumbent and the validation-selected beam
search. Its CLI rejects overlap with any original development ID and verifies
the supplied acquisition protocol matches the frozen selection, checkpoint,
and cohort size. No search setting, weight, or threshold was changed after the
fresh results were observed. These 50 images are now an observed test cohort;
future tuning must not treat them as a new untouched holdout.

## Actual results, October 19, 2026

The checkpoint remained:
`artifacts/cnn_experiments/boundary_v1/ranking/best.pt`, SHA-256
`2537358f0d51b6a3b25a65030b279a586601fdbead3570c837ba8e6baa0e0d79`.

| Cohort/method | Exact puzzles | Piece position accuracy | Correct assembled borders |
| --- | ---: | ---: | ---: |
| Validation: previous swaps | 64/75 (85.33%) | 89.71% | 96.53% |
| Validation: group descent | 67/75 (89.33%) | 91.04% | 97.40% |
| Validation: beam | 68/75 (90.67%) | 91.20% | 97.73% |
| **Fresh holdout: previous swaps** | **42/50 (84%)** | **86.80%** | **96.30%** |
| **Fresh holdout: frozen beam** | **45/50 (90%)** | **90.00%** | **97.05%** |

Validation beam recovered five previously failed puzzles and regressed one.
On the fresh cohort it recovered five and regressed two: a net three additional
exact puzzles, or six percentage points. Both fresh methods had the same 96.8%
local neighbor accuracy because they use identical frozen CNN scores. This is an
assembly improvement, not a new gain in the CNN's neighbor predictions.

The fresh gain is encouraging but small: seven puzzles changed exactness in
opposite directions, and only 50 images were evaluated. It does not establish
a statistically reliable universal improvement or prove that the CNN cannot
overfit. Five fresh puzzles remain incorrect. Some wrong arrangements still
receive better scores than ground truth, particularly with ambiguous backgrounds;
search alone cannot repair a scoring objective that prefers the wrong solution.
Border-aware/context model changes were not introduced in this conservative run.

The original 68/75 test result is unchanged. Do not compare it directly with
45/50: those are different image cohorts. The fresh comparison image shows the
first two recoveries and both regressions, rather than showing successes alone.

## Commands and outputs

Commands actually executed, in this order:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.search_experiments
.\.venv\Scripts\python.exe -m jigsaw_ai.fresh_holdout
.\.venv\Scripts\python.exe -m jigsaw_ai.search_experiments --source data/fresh_puzzles_v1 --output artifacts/search_experiments/fresh_audit_v1 --frozen-validation artifacts/search_experiments/validation_v1 --holdout-protocol data/fresh_holdout_v1/protocol.json
```

New output directories must be empty. To repeat acquisition or evaluation, choose
new explicit output paths and pass the corresponding frozen validation path;
existing results are protected. Acquisition does not resume partially populated
directories automatically.

- `artifacts/search_experiments/validation_v1/`: frozen configuration, all three
  per-puzzle reports, selected predictions, aggregate comparison and transition IDs.
- `data/fresh_holdout_v1/`: protocol, source files, license evidence, manifest,
  rejection reasons, split membership, and acquisition summary. Cache files can
  include fetched candidates outside the accepted 50; the manifest defines membership.
- `data/fresh_puzzles_v1/`: 50 verified puzzles, 1,250 lossless pieces, and index.
- `artifacts/search_experiments/fresh_audit_v1/`: checkpoint/protocol hashes,
  per-puzzle incumbent/beam reports, all 50 beam predictions, summary,
  `cohort_audit.json`, and `comparison.png` with recoveries and regressions.

Use the optional new search with an independent folder of 25 upright RGB
64 x 64 PNG pieces:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.solve_boundary --pieces PATH_TO_PIECES --checkpoint artifacts/cnn_experiments/boundary_v1/ranking/best.pt --output artifacts/my_beam_result --search beam
```

`--search` supports `greedy`, `swaps`, `groups`, and `beam`. The existing `--refine`
flag retains its swap behavior, and the original default remains greedy. Method
settings are saved with the reconstructed image and filename arrangement.
No additional dependency, Git commit, or publication was introduced.

## Verification

The final full suite passed **87 tests in 22.01 seconds**. The focused suites
passed **13 tests in 6.68 seconds**, including the added protocol mismatch guard.
Tests prove that group moves escape a controlled cycle where swaps are stuck,
preserve every piece, are repeatable, and match scalar border costs. They also
cover incumbent selection on ties, independent inference, reused-development
image rejection, fresh evaluation, output protection, difference hashes, and
protocol mismatch rejection. Real cohort auditing confirmed zero ID, known
photographer-profile, source URL, file checksum, or decoded-pixel overlap, and
the unchanged checkpoint. `pip check` and `git diff --check` passed.
