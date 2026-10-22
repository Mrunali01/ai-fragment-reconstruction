
# Jigsaw AI

Reconstruct shuffled image fragments using a trained CNN or classical edge
matching. Built with Python, PyTorch, FastAPI, PostgreSQL, Next.js, TypeScript,
and Tailwind CSS.

## Features

- Upload fragments or try a sample puzzle.
- Reconstruct images with background job tracking.
- Move pieces, undo changes, save corrections, and reconstruct again.
- Browse history, compare attempts, and export PNG, JSON, CSV, or ZIP results.

## Setup

Requires **Python 3.11+**. Run these commands from the project folder in Windows
PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/setup_tools.py
```

The setup script downloads local Node.js and PostgreSQL runtimes into `.tools`.

## Run

Start the backend and database:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-backend.ps1
```

In another terminal, open the same project folder and start the frontend:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-frontend.ps1
```

- **App:** http://127.0.0.1:3000
- **API docs:** http://127.0.0.1:8000/docs

The frontend script installs npm dependencies on its first run. Press **Ctrl+C**
in each terminal to stop the web servers. Stop the database separately with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop-database.ps1
```

## Usage and limitations

Choose **Try a sample puzzle**, or upload exactly **25 upright RGB PNG/JPEG
fragments**, each **64 × 64 pixels** and at most **256 KiB**. Click
**Reconstruct puzzle**, then use the interactive grid to inspect or correct it.

The app currently supports complete **5 × 5 square puzzles**. It does not support
traditional interlocking pieces, rotated or missing pieces, or uploading one
intact photo through the frontend. Reconstruction is not guaranteed to be correct.

Classical matching works without a trained checkpoint. The CNN requires the local
checkpoint at `artifacts/cnn_experiments/boundary_v1/ranking/best.pt`, or a path
configured through `JIGSAW_CHECKPOINT`. Checkpoints are not included in Git.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

PostgreSQL integration tests require `JIGSAW_TEST_DATABASE_URL`; otherwise those
tests are skipped. See [database setup](docs/database.md) and
[frontend documentation](docs/frontend.md) for additional checks.

See [model experiments](docs/cnn_experiments.md) and
[evaluation](docs/evaluation.md) for training details and measured results.
