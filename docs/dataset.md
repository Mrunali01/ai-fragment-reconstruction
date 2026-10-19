# Phase 1: intact image dataset

## Selection

We selected a pilot subset of **Open Images V7**, using its official image-ID
metadata linked from https://storage.googleapis.com/openimages/web/download_v7.html.
The image-ID CSVs predate V7; this is intentional because V7 publishes these same
image lists. Object labels and bounding boxes are unnecessary for our task.

| Candidate | Reason considered | Decision |
| --- | --- | --- |
| Open Images | Diverse natural scenes; selective image downloads; author and source metadata | Selected |
| COCO | Diverse scenes, but image rights remain with Flickr creators | No simpler licensing route than Open Images |
| DTD | Texture images, 300–640 pixels; research-purpose access | Repetitive textures are useful later, but narrow for the initial collection |
| Kaggle Landscape Pictures | Potentially convenient scene-image archive | Underlying photo permissions and dimensions were not verified |

References:

- Open Images description and license caveat: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- COCO terms: https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm
- DTD: https://robots.ox.ac.uk/~vgg/data/dtd/
- Kaggle alternative: https://www.kaggle.com/datasets/arnaud58/landscape-pictures

## Licensing

Open Images lists image licenses as CC BY 2.0, but asks users to verify each photo.
The acquisition script accepts only metadata listing that license and an original
Flickr page whose JSON-LD `ImageObject` lists the same license, identifies the same
photo page, and supplies an author. Removed, inaccessible, or changed-license
photos are rejected. The saved structured evidence includes verification time
and a hash of the fetched page. It does not constitute a guarantee of ownership.

Each manifest entry preserves the source URL, author, title, license URL, and
evidence path. Use these records for attribution when sharing images or derived
puzzles. Mark cropping, resizing, or other modifications when performed later.
The license summary is https://creativecommons.org/licenses/by/2.0/.

## Reproducible pilot selection

Default configuration: seed 42, minimum decoded side 320 pixels, targets of
350 training / 75 validation / 75 test images. Source split boundaries are
preserved; images are never moved from source test to project training.

To avoid transferring the enormous training metadata file, we snapshot the first
4,000 rows of each official split CSV and order this bounded pool by SHA256 of
`seed:source_split:image_id`. This is **not a uniform sample of the full dataset**.
Eligibility depends on currently accessible license evidence, dimensions, and
duplicates. The pilot subset therefore has selection bias and is intended to
validate the pipeline, not establish broad benchmark performance.

Existing attempt records, metadata snapshots, and images are reused on restart.
Failures are also cached so a later rerun preserves the selection. Use a new
output directory to retry eligibility against changed source websites. Rerunning
the same settings against retained snapshots and attempts reproduces membership;
fresh network acquisition cannot guarantee identical results as websites change.

Images are downloaded from the official CVDF S3 mirror and kept byte-for-byte as
downloaded. These are intact dataset images, which may already have been resized
by the dataset distributor; they are not guaranteed full-resolution camera
originals. No puzzle cropping or fragment generation occurs in this phase.

The validator verifies and fully decodes each file. Dimensions and decoded pixel
hashes use EXIF-corrected RGB pixels; future preprocessing must use the same EXIF
handling. The source `Rotation` field is retained separately, without silently
rotating the downloaded image. Future preprocessing must resolve its relationship
to the mirror's orientation rather than apply a second rotation blindly.

Exact file, decoded-pixel, and source-photo duplicates are excluded across all
splits. Near-duplicate detection is limited to contact-sheet inspection; resized,
cropped, or recompressed similar photos may evade exact hashes.

## Commands

Run from the project root with the virtual environment already created:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m jigsaw_ai.acquire_images
.\.venv\Scripts\python.exe -m jigsaw_ai.audit_dataset --previews
```

The downloader uses four network workers, 20-second request timeouts, at most
three attempts for transient failures, and response-size limits. No Kaggle account
or cloud credentials are needed. `--help` lists options. A new configuration needs
a new `--output` directory to avoid overwriting a different collection.

## Local artifacts

Everything below lives in Git-ignored `data/open_images/`:

- `images/`: intact image files with stable image-ID filenames.
- `metadata/`: official CSV candidate snapshots and their source URLs.
- `licenses/`: source-page structured license evidence for each checked photo.
- `attempts/`: per-candidate results for resumable acquisition.
- `manifest.jsonl`: one accepted image record per line, including hashes.
- `splits.json`: image IDs assigned before any fragment generation.
- `failures.json`: rejected candidates and reasons.
- `config.json`: reproducibility settings.
- `acquisition_summary.json`: counts and accepted image bytes.
- `audit.json`: verified counts, dimensions, duplicates, and disk usage.
- `previews/`: contact sheets labeled with image IDs, grouped by split.

Only paths in the manifest are dataset members. A downloaded duplicate rejected
after comparison can remain in the cache; never glob `images/` as training input.
Network interruption can leave a partial manifest; the audit requires all target
counts before reporting success. Keep the manifest and cached records to reproduce
the exact subset. No data or license evidence is pushed to GitHub automatically.

## Verified results

Acquisition completed October 14, 2026; final checks ran October 15, 2026.

| Check | Actual result |
| --- | --- |
| Accepted images | 500 |
| Source-preserving splits | 350 training / 75 validation / 75 test |
| Accepted image bytes | 171,997,581 (about 172 MB or 164 MiB) |
| Complete local dataset directory | About 186 MB, including metadata, evidence, previews, and cache |
| Width range after EXIF orientation | 576–6,016 pixels |
| Height range after EXIF orientation | 454–4,000 pixels |
| Exact file, decoded-pixel, and source-URL duplicates in manifest | 0 |
| Rejected candidates | 144 |
| Automated tests | 13 passed |
| Dependency consistency | `pip check`: no broken requirements |
| Actual restart | Manifest SHA256 unchanged |
| Contact sheets | 11 generated and visually inspected |

Width and height extrema are independent; they are not the dimensions of a
single image. No training or reconstruction performance is measured in Phase 1.

Rejections: 106 HTTP 404 responses, 13 HTTP 403 responses, 15 original pages no
longer listing CC BY 2.0, 5 unsupported source URLs, and 5 pages without matching
image-specific structured license evidence. The rejection log does not claim that
all HTTP errors originated at the same endpoint.

The image cache contains 502 files, while the manifest selects exactly 500.
Two files remain from earlier candidate selection; they are not dataset members.
Use the manifest, not a directory glob, in subsequent phases.

Contact-sheet review found diverse scenes, historical images, graphics, and some
blurry or dark images. Some images are sideways or upside down even after EXIF
handling; their source rotation metadata is retained for Phase 2 to resolve.
No obvious duplicate pair was identified during this limited visual inspection;
this does not establish absence of all near-duplicates.

Verification initially encountered a transient Windows permission error during
atomic manifest replacement. A bounded retry now tolerates brief file locks;
a test verifies old output is preserved until replacement succeeds. Persistent
permission errors still stop the operation. Run only one acquisition process per
output directory; concurrent writers to the same directory are not supported.
