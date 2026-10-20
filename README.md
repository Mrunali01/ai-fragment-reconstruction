
<<<<<<< HEAD
AI-Powered Jigsaw Puzzle Solver: a project built incrementally to reconstruct
images from shuffled fragments using computer vision, deep learning, and optimization.

## Current status

Phase 0 provides the Python package, development environment, and test setup.
Phase 1 adds Open Images acquisition, per-photo license evidence, intact-image
validation, split manifests, and dataset auditing. See [dataset documentation](docs/dataset.md).
Phase 2 adds reproducible 5 × 5 puzzle generation and ground-truth verification.
See [fragment documentation](docs/fragments.md).
Phase 3 adds balanced fragment pairs, direction labels, a lazy image loader,
and independent label auditing. See [pair documentation](docs/pairs.md).
Phase 4 adds classical edge scoring, greedy reconstruction, and held-out evaluation.
See [baseline documentation](docs/baseline.md).
Phase 5 adds an untrained Siamese CNN, tensor preparation, and masked losses.
See [model documentation](docs/model.md).
Phase 6 adds bounded CPU training, validation-selected checkpoints, learning
curves, and held-out pair evaluation. See [training documentation](docs/training.md).
Phase 7 connects frozen CNN scores to greedy puzzle assembly, saves reconstructed
images, and measures position, border, and exact reconstruction accuracy.
See [reconstruction documentation](docs/reconstruction.md).
Additional approved CNN experiments add a boundary Siamese model, hard-negative
and ranking training, hybrid scoring, and swap refinement. See
[CNN experiments](docs/cnn_experiments.md) for methodology, measured results,
and the standalone reconstruction command.
The trained boundary CNN with swap refinement solved 68/75 existing test puzzles
exactly, compared with 0/75 for the original CNN and 63/75 for the classical
baseline. Settings were selected on validation; this test set was previously
inspected and does not replace a fresh final evaluation.
An additional frozen-model search experiment improved exact reconstruction on
50 fresh, development-disjoint images from 42/50 to 45/50. See
[search improvements and holdout protocol](docs/search_improvements.md), including
the regressions and sample-size limitations.
Phase 8 adds a shared image-only pipeline, validation-calibrated pair evaluation,
six classical/CNN search comparisons, raw prediction exports, uncertainty, and
failure analysis. See [end-to-end evaluation](docs/evaluation.md).
Phase 9 adds a FastAPI service for fragment uploads, validated reconstruction,
and retrieval of saved arrangements and PNG results. See [API documentation](docs/api.md).
Phase 10 adds PostgreSQL puzzle, job and result metadata with images on disk.
See [database setup](docs/database.md).
Phase 11 adds a responsive Next.js, TypeScript and Tailwind frontend for fragment
uploads, solver selection and viewing saved reconstructions. See [frontend setup](docs/frontend.md).
Phases 12–17 add an interactive puzzle grid, saved human correction versions,
bounded background jobs, relative edge evidence, reconstruction history and
comparison, an analytics API, and PNG/JSON/CSV/ZIP downloads. See the
[implementation and verification report](docs/phases_12_17.md).
After moving or shuffling tiles, **Reconstruct again** runs a new attempt with
the same pieces, without another upload. **Restore solver result** immediately
returns to the existing arrangement. Save manual corrections before rerunning
if you want to keep that layout; a new attempt does not guarantee improvement.

## Run the web application (Windows PowerShell)

From the project root, install the local development runtimes once:

```powershell
.venv/Scripts/python.exe scripts/setup_tools.py
.venv/Scripts/python.exe -m pip install -e '.[dev]'
```

Start the backend and local PostgreSQL database in one terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-backend.ps1
```

Start the frontend in another terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-frontend.ps1
```

Open **http://127.0.0.1:3000**. Choose **Try a sample puzzle**, then
**Reconstruct puzzle**, or upload 25 upright RGB 64 × 64 PNG/JPEG fragments.
The sample is attributed CC BY 2.0 imagery; its license and modifications are
included in `frontend/public/demo/attribution.json`. The CNN requires the existing
local trained checkpoint; classical matching also works without that checkpoint.
Press Ctrl+C in each terminal to stop the web servers. Stop PostgreSQL separately
with `powershell -ExecutionPolicy Bypass -File scripts/stop-database.ps1`.

## Setup (Windows PowerShell)

Use Python 3.11 or newer. The initial setup was verified with Python 3.13.2.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

Calling the virtual environment's Python directly avoids needing to activate it
or change PowerShell execution policy. The editable install makes changes under
`src/` available immediately. Run these commands from the project root.

On Linux or macOS, use `.venv/bin/python` instead of
`.\.venv\Scripts\python.exe`.

## Layout

```text
src/jigsaw_ai/       Python package; future pipeline modules belong here
tests/              Automated tests
pyproject.toml      Package metadata, dependencies, and pytest configuration
```

Pillow validates images and creates contact sheets. NumPy prepares pixel arrays;
PyTorch provides the Siamese CNN and losses; Matplotlib plots learning curves.
For the CPU installation command, see
[model setup](docs/model.md). The development extra installs pytest; pip also
installs setuptools to build the package. Dependencies are added only when needed.

Generated datasets, outputs, checkpoints, and local secrets are ignored by Git.
Do not store API keys or downloaded image datasets in source control.

## Environment inspection

Inspected on October 13, 2026:

- Python: 3.13.2; Git: 2.46.2.windows.1.
- Node.js, npm, uv, and Poetry were not found on PATH. Node.js is not needed yet.
- Windows reports Intel Iris Xe graphics. NVIDIA management and CUDA compiler
  tools were not found; plan for CPU execution initially.
- PyTorch was deferred in Phase 0; Phase 5 now adds CPU computation checks.

## Phase 0 verification

- Editable package installation with the development extra succeeded.
- `.\.venv\Scripts\python.exe -m pytest`: 1 test passed.
- `.\.venv\Scripts\python.exe -m pip check`: no broken requirements found.
- Git ignore checks confirmed that the virtual environment, package metadata,
  dataset files, and output files are excluded. No Git remote is configured.

The sandbox process launcher failed during verification, so checks ran outside
the sandbox with approval. Git reported different repository ownership between
the sandbox and Windows accounts; read-only checks used
`git -c safe.directory=D:/ai-fragment-reconstruction ...` for this workspace.
No global Git configuration was changed.

## Development workflow

Each phase requires explicit approval before implementation. Run the relevant
tests after changes, document actual results, and review Git changes before
committing. No commit or remote repository is created automatically.

Generate the approved Phase 2 puzzles locally:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.generate_puzzles
```

Generate and audit the approved Phase 3 pair data:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.generate_pairs
.\.venv\Scripts\python.exe -m jigsaw_ai.audit_pairs
```

Run the approved classical baseline:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_baseline --split validation
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_baseline --split test
```

Check the approved Phase 5 model without training:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.inspect_model
```

Train and evaluate the approved Phase 6 model:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.train
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_model --split validation
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_model --split test
```

Training output must be new/empty. Reconstruct the approved Phase 7 puzzles:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_reconstruction --split validation
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_reconstruction --split test
```

Run the approved Phase 8 evaluation:

```powershell
.\.venv\Scripts\python.exe -m jigsaw_ai.evaluate_pipeline
```

Report output must be new/empty; use `--output` for a separate reproduction.
Start the approved Phase 9 API from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn jigsaw_ai.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to upload 25 fragments. The service uses the
existing local checkpoint and stores results under `artifacts/api_results`.
Dataset split assignments are preserved to prevent leakage. Phases 0–17 are
implemented. Phase 18 (Docker and deployment) requires separate approval.
=======
>>>>>>> 7049350eb76aa1a2bcc02020438f619545b7726a
