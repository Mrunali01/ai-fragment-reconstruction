# Phases 10 and 11 completion report

## A. Changes made

Phase 10 adds real PostgreSQL persistence with plain parameterized SQL and
Psycopg. Three linked tables store puzzle input metadata, job state and completed
results. Images remain on disk. Startup validates schema version 1; configured
database outages fail explicitly. Filesystem-only mode remains available for
existing CLI/API users and tests.

Phase 11 adds a working Next.js/TypeScript/Tailwind web application. Users can
upload exactly 25 square fragments or load an attributed sample, inspect previews,
select CNN/classical scoring and assembly search, submit a reconstruction, view
or save the PNG, and reload its persisted result URL. Loading, validation, offline,
model-unavailable and busy states have browser coverage.

## B. Files created or modified

| File | Purpose |
| --- | --- |
| `src/jigsaw_ai/database.py` | PostgreSQL schema, parameterized queries and transactions. |
| `src/jigsaw_ai/api.py` | Optional configured PostgreSQL store, job transitions, outage handling and health checks. |
| `tests/test_database.py` | Real PostgreSQL persistence/restart/failure tests with isolated temporary schemas. |
| `pyproject.toml` | Adds Psycopg binary runtime dependency. |
| `.gitignore` | Excludes local runtimes, database cluster, npm/build/test output and TypeScript caches. |
| `scripts/setup_tools.py` | Downloads local official Windows Node/PostgreSQL runtimes. |
| `scripts/start-database.ps1` | Initializes SCRAM-protected loopback PostgreSQL and creates the development database. |
| `scripts/stop-database.ps1` | Stops that local cluster. |
| `scripts/start-backend.ps1` | Starts PostgreSQL, configures the API connection and runs FastAPI. |
| `scripts/start-frontend.ps1` | Uses local Node, installs locked dependencies if needed and starts Next.js development. |
| `frontend/package.json` | Frontend runtime/dev dependencies and scripts. |
| `frontend/package-lock.json` | Exact npm dependency resolutions. |
| `frontend/tsconfig.json` | Strict TypeScript configuration. |
| `frontend/next-env.d.ts` | Next-generated framework type references. |
| `frontend/next.config.ts` | Same-origin API proxy rewrites. |
| `frontend/postcss.config.mjs` | Tailwind PostCSS integration. |
| `frontend/.env.example` | Documents configurable backend proxy URL. |
| `frontend/app/layout.tsx` | Root document, metadata and CSS entry. |
| `frontend/app/page.tsx` | Upload workspace, state handling, previews, settings and result retrieval. |
| `frontend/app/globals.css` | Responsive visual styling, focus and reduced-motion support. |
| `frontend/lib/api.ts` | Typed API contracts, readable errors and browser input checks. |
| `frontend/playwright.config.ts` | Chromium tests against a production frontend. |
| `frontend/tests/workspace.spec.ts` | Validation, mobile, failure/retry and real reconstruction browser checks. |
| `frontend/public/demo/attribution.json` | Source, creator, CC BY 2.0 license and transformations for demo assets. |
| `README.md` | Updated phase status and two-terminal application startup. |
| `docs/database.md` | Database setup, storage, transactions, limitations and verification. |
| `docs/frontend.md` | Frontend usage, architecture, configuration, licensing and tests. |
| `docs/api.md` | Cross-links current PostgreSQL/frontend behavior and outage responses. |
| `docs/phases_10_11.md` | This completion report. |
| `frontend/public/demo/piece_000.png` | Shuffled, RGB 64 ? 64 sample fragment 1; licensed as described in attribution.json. |
| `frontend/public/demo/piece_001.png` | Shuffled, RGB 64 ? 64 sample fragment 2; licensed as described in attribution.json. |
| `frontend/public/demo/piece_002.png` | Shuffled, RGB 64 ? 64 sample fragment 3; licensed as described in attribution.json. |
| `frontend/public/demo/piece_003.png` | Shuffled, RGB 64 ? 64 sample fragment 4; licensed as described in attribution.json. |
| `frontend/public/demo/piece_004.png` | Shuffled, RGB 64 ? 64 sample fragment 5; licensed as described in attribution.json. |
| `frontend/public/demo/piece_005.png` | Shuffled, RGB 64 ? 64 sample fragment 6; licensed as described in attribution.json. |
| `frontend/public/demo/piece_006.png` | Shuffled, RGB 64 ? 64 sample fragment 7; licensed as described in attribution.json. |
| `frontend/public/demo/piece_007.png` | Shuffled, RGB 64 ? 64 sample fragment 8; licensed as described in attribution.json. |
| `frontend/public/demo/piece_008.png` | Shuffled, RGB 64 ? 64 sample fragment 9; licensed as described in attribution.json. |
| `frontend/public/demo/piece_009.png` | Shuffled, RGB 64 ? 64 sample fragment 10; licensed as described in attribution.json. |
| `frontend/public/demo/piece_010.png` | Shuffled, RGB 64 ? 64 sample fragment 11; licensed as described in attribution.json. |
| `frontend/public/demo/piece_011.png` | Shuffled, RGB 64 ? 64 sample fragment 12; licensed as described in attribution.json. |
| `frontend/public/demo/piece_012.png` | Shuffled, RGB 64 ? 64 sample fragment 13; licensed as described in attribution.json. |
| `frontend/public/demo/piece_013.png` | Shuffled, RGB 64 ? 64 sample fragment 14; licensed as described in attribution.json. |
| `frontend/public/demo/piece_014.png` | Shuffled, RGB 64 ? 64 sample fragment 15; licensed as described in attribution.json. |
| `frontend/public/demo/piece_015.png` | Shuffled, RGB 64 ? 64 sample fragment 16; licensed as described in attribution.json. |
| `frontend/public/demo/piece_016.png` | Shuffled, RGB 64 ? 64 sample fragment 17; licensed as described in attribution.json. |
| `frontend/public/demo/piece_017.png` | Shuffled, RGB 64 ? 64 sample fragment 18; licensed as described in attribution.json. |
| `frontend/public/demo/piece_018.png` | Shuffled, RGB 64 ? 64 sample fragment 19; licensed as described in attribution.json. |
| `frontend/public/demo/piece_019.png` | Shuffled, RGB 64 ? 64 sample fragment 20; licensed as described in attribution.json. |
| `frontend/public/demo/piece_020.png` | Shuffled, RGB 64 ? 64 sample fragment 21; licensed as described in attribution.json. |
| `frontend/public/demo/piece_021.png` | Shuffled, RGB 64 ? 64 sample fragment 22; licensed as described in attribution.json. |
| `frontend/public/demo/piece_022.png` | Shuffled, RGB 64 ? 64 sample fragment 23; licensed as described in attribution.json. |
| `frontend/public/demo/piece_023.png` | Shuffled, RGB 64 ? 64 sample fragment 24; licensed as described in attribution.json. |
| `frontend/public/demo/piece_024.png` | Shuffled, RGB 64 ? 64 sample fragment 25; licensed as described in attribution.json. |

## C. Beginner-friendly walkthrough

`PostgresStore.start()` inserts puzzle metadata and a running job together.
After the existing image-only pipeline finishes, the API writes the PNG and JSON
copy to disk. `complete()` inserts result metadata and marks the job completed
inside one PostgreSQL transaction. A transaction makes these database changes
succeed together or roll back together. Handled failures mark the job failed;
if database completion fails, the new image directory is cleaned up. Database
URLs and generated passwords remain local, outside Git.

The browser owns selected `File` objects and preview URLs. `validateFiles()` checks
basic input mistakes before upload; FastAPI independently checks actual content.
Submitting creates multipart `FormData`, which Next.js forwards to FastAPI. The
backend runs the existing frozen model and solver. React renders the returned
PNG and updates the URL with its ID. Reloading fetches completed metadata from
PostgreSQL. Original uploads are intentionally not persisted, so previews clear
on reload while the completed result remains available.

## D. Commands and actual results

- `.venv/Scripts/python.exe -m pip install -e '.[dev]'`: succeeded.
- `powershell -ExecutionPolicy Bypass -File scripts/start-database.ps1`: initialized
  and started PostgreSQL 18.6 on 127.0.0.1:55432 using SCRAM authentication.
- `.venv/Scripts/python.exe -m pytest -q` with `JIGSAW_TEST_DATABASE_URL` set:
  **114 passed in 82.81 seconds**, no skipped tests in that run.
- `npm.cmd --prefix frontend install`: succeeded; npm reported zero vulnerabilities.
- `npm.cmd --prefix frontend run build`: optimized production build passed.
- `npm.cmd --prefix frontend run typecheck`: passed.
- `npm.cmd --prefix frontend run test:e2e` with `JIGSAW_LIVE_TEST=1`:
  **4 passed in 9.2 seconds**, including the actual trained-CNN upload and page reload.
- Initial browser run had two selector failures because Next.js adds its own
  route-announcement alert. Scoping error assertions to the main application
  fixed those test selectors; all four final browser tests passed.
- Browser runner printed harmless FORCE_COLOR/NO_COLOR environment warnings.
- `pip check`: no broken requirements. `git diff --check`: passed, with Git's
  existing Windows LF-to-CRLF notices.
- A real HTTP check through port 3000 returned 200 for the page, proxied health,
  persisted metadata and PNG. PostgreSQL contains the same completed response,
  25 input metadata entries and a relative image reference.
- The browser-created PNG matches the prior frozen pipeline prediction byte for
  byte, and the checkpoint SHA-256 matches its recorded value. No retraining,
  evaluation selection or new performance claim was introduced.

Evidence: ignored `artifacts/frontend/verification.json`, `live-result.json`,
`desktop-result.png` and `mobile.png`. Screenshots were inspected visually.

## E. Decisions and limitations

Plain Psycopg SQL is sufficient for three tables; no ORM or task queue is needed.
The schema initializes idempotently at version 1; future schema changes need
explicit migrations. Existing filesystem results are not automatically imported.
A crash can leave a running job or orphaned files because PostgreSQL and the
filesystem do not share one transaction. Back up both database and result files.

This remains a local application for complete, upright 5 ? 5 puzzles of RGB
64 ? 64 fragments. It does not guarantee correct reconstruction. Originals are
not retained, and users must reselect them for another attempt after reloading.
There is no history listing, manual movement/correction, background job queue,
confidence overlay, authentication or deployment yet. The sample is an already
observed evaluation image, not fresh evidence of generalization.

## F. Git status

Changes are implemented and checked, ready for user review before committing.
No staging, commit, push or publication was performed. Prior uncommitted Phase 8
and Phase 9 files were preserved. Local runtime archives, database/password,
trained artifacts, npm dependencies and build outputs are ignored.

## G. Next phase

Phase 12 adds the interactive reconstruction canvas: selectable pieces, movement
and clear interaction states. Saved human corrections belong to Phase 13.
Phase 12 has not started and requires separate approval.
