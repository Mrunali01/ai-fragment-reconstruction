# Phase 10: PostgreSQL persistence

The API stores three related records: `puzzles` contains the 25 input filenames and
SHA-256 hashes, `reconstruction_jobs` records solver, search, timestamps and status,
and `reconstruction_results` stores the completed response and relative PNG path.
No image bytes go into PostgreSQL. Files remain under `JIGSAW_RESULTS_DIR` (default
`artifacts/api_results`). Back up the database **and** that directory together.

## Local Windows development

```powershell
.venv/Scripts/python.exe scripts/setup_tools.py
.venv/Scripts/python.exe -m pip install -e '.[dev]'
powershell -ExecutionPolicy Bypass -File scripts/start-backend.ps1
```

The helper starts a local PostgreSQL 18.6 instance bound to `127.0.0.1:55432`,
creates `jigsaw_ai`, and launches FastAPI on port 8000. It uses SCRAM authentication
with a generated local password in ignored `.local/db-password`. Binaries live in
ignored `.tools`, database files in ignored `.local/postgres`. No Windows service,
global PATH entry or system installation is created. PostgreSQL keeps running
after the backend stops; stop it explicitly with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/stop-database.ps1
```

For an existing PostgreSQL server, set `JIGSAW_DATABASE_URL` to its connection URL
before running Uvicorn. The role needs permission to create tables on first run.
Schema version 2 adds `workspace_records` for append-only human correction
versions and persistent background job states. Existing version 1 databases
upgrade transactionally at startup; repeated initialization is idempotent. Future
schema changes need an explicit migration. A configured unreachable database
prevents startup. Unset the variable to retain Phase 9 filesystem-only behavior;
there is no automatic fallback when a configured database fails. Old filesystem
results are not automatically imported into PostgreSQL.

Each accepted, decoded request starts a `running` job, then writes result files
and commits result metadata plus `completed` status in one database transaction.
Handled solver/storage errors mark it `failed`; a database completion error removes
the newly created result files. PostgreSQL and the filesystem cannot share one
atomic transaction: a process crash can leave a `running` job or orphaned files.
Background job state now survives restart; interrupted queued/running work is
marked failed with an explanation, not resumed. Correction images live under
`{result_id}/corrections/{correction_id}.png`; their version metadata is stored in
PostgreSQL. History and analytics read stored completed results. One request
creates one puzzle and one job; retrying creates a new ID. Originals are processed
in memory and not retained. See [Phases 12–17](phases_12_17.md) for the local worker
and cross-filesystem transaction limitations. Use one API worker per result root.

## Verification

```powershell
$dbPassword = (Get-Content .local/db-password -Raw).Trim()
$env:JIGSAW_TEST_DATABASE_URL = "postgresql://jigsaw:${dbPassword}@127.0.0.1:55432/jigsaw_ai"
.venv/Scripts/python.exe -m pytest tests/test_database.py tests/test_api.py -q
```

Integration tests create a unique temporary PostgreSQL schema and remove that schema
afterward. They verify actual database round trips, retrieval after app restart,
failed jobs and storage cleanup, plus explicit failure on an unreachable database.
Without the test URL, the two live-database tests are skipped explicitly.

Runtime downloads use official Node.js and EDB HTTPS endpoints. Node's archive is
checked against its published SHA-256; the PostgreSQL binary ZIP uses EDB TLS and
ZIP integrity checks, without a separately published checksum verification here.
