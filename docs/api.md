# Phase 9: FastAPI backend

The API wraps the existing frozen solver. It accepts independently supplied
fragments, reconstructs them within the request, and returns an arrangement and
a retrievable PNG. It does not train the model or calculate ground-truth accuracy
for uploads. This phase adds HTTP access, validation, local result storage, and
tests. Phase 10 now adds [PostgreSQL persistence](database.md), and Phase 11 adds
the [Next.js frontend](frontend.md). The commands below still support the original
filesystem-only API mode when `JIGSAW_DATABASE_URL` is unset.

## Install and run

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m uvicorn jigsaw_ai.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive upload form. Select
`POST /reconstructions`, choose Try it out, attach 25 fragments under `pieces`,
and execute. Follow the returned `image_url` on the same API server.
`/openapi.json` exposes the generated request/response contract.

The trained checkpoint is loaded once at startup from:
`artifacts/cnn_experiments/boundary_v1/ranking/best.pt`.
It must exist locally; ignored checkpoints are not assumed to arrive with a Git
clone. If it is missing or invalid, health reports CNN unavailable and CNN
requests return 503. Classical requests still work when explicitly selected.
The API never silently substitutes a different scorer.

Optional environment configuration, set before starting the server:

```powershell
$env:JIGSAW_CHECKPOINT = "D:\ai-fragment-reconstruction\artifacts\cnn_experiments\boundary_v1\ranking\best.pt"
$env:JIGSAW_RESULTS_DIR = "D:\ai-fragment-reconstruction\artifacts\api_results"
```

These are process environment variables; the API does not automatically read a
`.env` file. The default results directory is `artifacts/api_results` relative
to the working directory. Use a single Uvicorn worker for this local CPU setup.

## Endpoints

| Method/path | Purpose | Successful response |
| --- | --- | --- |
| `GET /health` | Service health and CNN/classical availability | 200 JSON |
| `POST /reconstructions` | Upload pieces and complete reconstruction | 201 JSON |
| `GET /reconstructions/{id}` | Retrieve saved metadata/arrangement | 200 JSON |
| `GET /reconstructions/{id}/image` | Download reconstructed image | 200 `image/png` |
| `GET /docs`, `GET /openapi.json` | Interactive documentation and schema | 200 |

`POST /reconstructions` uses **multipart/form-data**, not a JSON body:

- `pieces`: exactly 25 repeated file parts. Each must decode as one RGB
  64 x 64 PNG or JPEG image. Content is checked rather than trusting filenames
  or declared MIME types. Grayscale, RGBA, animated/multiframe images, unsupported
  formats, and incorrect dimensions are rejected; no implicit resize or rotation
  is performed. Pieces must already be upright.
- `scorer`: `cnn` (default) or `classical`.
- `search`: `beam` (default), `groups`, `swaps`, or `greedy`.

Each file is limited to 256 KiB, and the complete request is limited to 8 MiB,
including multipart headers. The body limit is checked before multipart parsing,
including requests without Content-Length/chunked bodies. Bodies within the
limit are buffered, which keeps this implementation straightforward for these
small puzzles; it is not a large-file streaming service.

The response contains a server-generated UUID, UTC creation timestamp, completed
status, method settings, fixed grid/piece/image dimensions, checkpoint SHA-256
for CNN results, and a row-major arrangement. Every placement includes its
zero-based row/column, original multipart `upload_index`, display filename,
and original upload byte checksum. Duplicate filenames are allowed; identify
pieces by upload index rather than assuming names are unique.

`mean_border_cost` is the actual solver objective. It is **not confidence or
accuracy**, and its units differ between scorers. No ground-truth accuracy is
invented. The output image is RGB 320 x 320, using every supplied fragment once.

## Example client

The development extra installs the HTTP client used by the current Starlette
TestClient. With the server running, this Python example uploads an existing
generated puzzle without submitting its ground truth:

```python
from pathlib import Path
import httpx2

pieces = Path("data/fresh_puzzles_v1/test/0ec65bf716d62105/pieces")
files = [("pieces", (p.name, p.read_bytes(), "image/png"))
         for p in sorted(pieces.glob("*.png"))]

with httpx2.Client(base_url="http://127.0.0.1:8000", timeout=60) as client:
    response = client.post("/reconstructions", files=files,
                           data={"scorer": "cnn", "search": "beam"})
    response.raise_for_status()
    result = response.json()
    image = client.get(result["image_url"])
    image.raise_for_status()
    Path("artifacts/api_example.png").write_bytes(image.content)
    print(result["id"], result["arrangement"])
```

## Errors and execution behavior

| Status | Meaning |
| --- | --- |
| 400 | Malformed request, including invalid Content-Length |
| 404 | Unknown result or missing result image |
| 413 | Per-file or total body limit exceeded |
| 422 | Missing/wrong file count, invalid image, unsupported method, or invalid UUID |
| 503 | CNN unavailable, or another reconstruction is using the worker |
| 500 | Inference/storage failure, or unreadable/corrupt saved metadata |

When busy, the API returns `Retry-After: 1`. The caller can retry the request;
there is no queued job or polling status masquerading as a background operation.
The POST response is returned only after computation and storage complete.
The CPU computation runs in a thread so health/metadata requests can remain
responsive. A nonblocking lock admits one reconstruction at a time per worker.
Multiple Uvicorn workers would have separate models and separate locks.

Unexpected server errors are logged internally while clients receive concise
messages. Results are saved under generated UUID directories, never upload
filenames. Display names are stripped of directory components/control characters.
Existing results are not overwritten, and handled storage failures clean up
the new partial result. Normal result files survive application restarts.

## Storage and scope

Each completed result contains:

```text
artifacts/api_results/{uuid}/
    image.png
    result.json
```

Original upload files are decoded in memory and closed after the request; they
are not retained by this API. Metadata retains their labels, input order, and
checksums. With `JIGSAW_DATABASE_URL` configured, PostgreSQL is authoritative for
completed metadata; PNGs stay in this directory. Database read/write outages
return 503, and a configured database must be reachable at startup. There is no
authentication or retention policy yet. This is a local
development API; deployment is planned for Phase 18.

Phases 12–17 now add correction versions, a bounded local background worker,
history, diagnostics, analytics and exports. The endpoint list and current
semantics are documented in [the feature report](phases_12_17.md). The original
synchronous endpoints above remain available. Background jobs do not use Redis.

`POST /reconstructions/{id}/reconstruct` accepts a JSON body with `order` (a strict
permutation of upload indices 0–24), and optional `scorer` / `search` overrides.
It recovers decoded pixels from the saved lossless mosaic, submits them in that
order, and returns 202 with a background job. The new result records
`source_result_id` and `source_order`. Filenames and file hashes are inherited as
provenance from the original uploads; these are not newly uploaded file bytes.
The endpoint shares queue limits and model availability checks with normal jobs.
The source result and its correction versions remain unchanged. A missing saved
image returns 404; an incompatible saved image returns 409.

## Code walkthrough

- `api.py` creates the application, loads the frozen CNN during lifespan startup,
  connects routes to the image-only pipeline, serializes computation, and stores
  completed images/metadata. `create_app` accepts explicit checkpoint/storage
  paths so tests use isolated directories rather than touching real results.
- `api_uploads.py` bounds raw request bytes before parsing, validates decoded
  fragment images, and treats supplied filenames as display labels.
- `api_schemas.py` defines typed health, placement, and reconstruction responses
  with bounded coordinates, valid checksum shape, and 25 placements.
- `pipeline.py` remains the tested ML/optimization implementation. Routes do
  not duplicate the solver, receive ground truth, or update model weights.

## Verification (October 19, 2026)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

The final full suite passed **110 tests in 67.77 seconds**, including **19 API
cases**, without test warnings or failures. Checks cover real CNN inference
against the direct pipeline, unchanged weights, exact piece use and PNG pixels,
file count and method validation, empty/truncated/invalid images, wrong dimensions
and modes, file/body limits including chunked bodies, unavailable/corrupt models,
duplicate/path-like filenames, result retrieval across restarts, unknown IDs,
storage cleanup, missing images, corrupt metadata, concurrent busy behavior,
responsive health during computation, and generated OpenAPI upload constraints.

A real localhost TCP smoke test started Uvicorn, uploaded all 25 real fragments
from `0ec65bf716d62105`, retrieved metadata and the image, and fetched `/docs`.
Health, metadata, image, and docs returned 200; reconstruction returned 201.
The returned PNG matched the prior frozen pipeline prediction byte-for-byte,
and the checkpoint hash was unchanged. The one warm-model request took
0.3066 seconds, including upload/inference/storage; this is not a latency benchmark
or a guarantee for other requests. The test server shut down cleanly afterward.
Evidence is saved in `artifacts/api_smoke/phase9_v1/summary.json`, `response.json`,
and its result directory.

Editable installation and `pip check` succeeded. One dependency download was
retried after a connection reset. An initial test-client deprecation warning was
resolved using the current supported HTTPX2 client instead of suppressing it.
The tested versions are FastAPI 0.148.0, Starlette 1.9.0, Pydantic 2.14.1,
Uvicorn 0.54.0, python-multipart 0.0.32, and HTTPX2 2.14.0. Runtime dependencies
and development-only client dependencies are declared separately in pyproject.

The documented ML limitations remain: complete 5 x 5 upright square puzzles only,
imperfect reconstruction, and ambiguous/identical pieces. No new accuracy claim
or model-tuning experiment was introduced by exposing the solver over HTTP.

Framework references: [FastAPI uploads](https://fastapi.tiangolo.com/tutorial/request-files/),
[FastAPI shared-model lifespan](https://fastapi.tiangolo.com/advanced/events/), and
[Starlette's current TestClient implementation](https://github.com/Kludex/starlette/blob/main/starlette/testclient.py).
