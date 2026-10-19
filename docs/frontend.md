# Phase 11: web frontend

The app is in `frontend/`, using Next.js App Router, React, TypeScript and
Tailwind CSS. The light green workspace has an upload panel, fragment previews,
solver settings and a result panel. It adapts to a one-column mobile layout.
No model weights or database credentials are sent to the browser.

## Start locally

Use the two terminal commands in the root README:

```powershell
# Terminal 1, project root: PostgreSQL + FastAPI
powershell -ExecutionPolicy Bypass -File scripts/start-backend.ps1
# Terminal 2, project root: Next.js
powershell -ExecutionPolicy Bypass -File scripts/start-frontend.ps1
```

Open http://127.0.0.1:3000, not port 8000. API docs remain at
http://127.0.0.1:8000/docs. The frontend launcher adds the local Node runtime to
its process PATH and runs `npm ci` if dependencies are absent. The lockfile
records exact npm package resolutions.

For system Node installations, use Node 24 LTS and run `npm ci` followed by
`npm run dev` inside `frontend`. A production build uses `npm run build`, then
`npm start`. Neither launcher changes the global Windows execution policy.

## Try a puzzle

1. Choose **Try a sample puzzle**, or select/drop exactly 25 PNG/JPEG files.
2. Files must each be 64 × 64 pixels, upright, RGB and at most 256 KiB.
3. Choose the trained CNN or classical edge matching and an assembly search.
4. Choose **Reconstruct puzzle**. Loading disables the inputs to avoid duplicate
   submissions; backend errors preserve the chosen files for retry.
5. View or save the reconstructed PNG. The URL contains its result ID; reloading
   retrieves saved metadata and the image. Original uploaded fragments are not
   retained across reloads, so the preview panel starts empty after a reload.

The browser checks count, filename extension, byte size and decoded dimensions.
FastAPI remains authoritative for content format, RGB mode, frame count and
request limits. PNG transparency or grayscale must be converted to RGB before
upload. The UI does not silently resize images or rotate pieces.

The sample consists of the already-observed evaluation image “Scratch” by Philo
Nordlund, [Flickr source](https://www.flickr.com/photos/philon/2461050066),
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). It was resized, center
cropped and sliced into 25 shuffled fragments. Its files include attribution;
the UI also displays credit. Ground-truth positions are not sent with uploads.
This sample is for demonstrating the UI, not new evidence of model generalization.

## How the components connect

`app/page.tsx` owns selected files and UI state. Object URLs make local previews
without uploading anything until reconstruction is requested, and are revoked
when files change. `lib/api.ts` holds shared response types, readable API error
handling and input checks. A `FormData` upload reaches the existing FastAPI
pipeline through `/api/reconstructions`; the response identifies the saved PNG.

`next.config.ts` proxies `/api/*` to `http://127.0.0.1:8000/*`, using Next.js
[external rewrites](https://nextjs.org/docs/app/api-reference/config/next-config-js/rewrites).
This keeps browser requests on the frontend origin. Set `JIGSAW_API_URL` in
`frontend/.env.local` for another trusted backend, then restart development or
rebuild production. Tailwind is loaded through its
[PostCSS integration](https://tailwindcss.com/docs/installation/framework-guides/nextjs).

## Checks

```powershell
$env:PATH = "$PWD/.tools/node-v24.22.0-win-x64;$env:PATH"
npm.cmd --prefix frontend run build
npm.cmd --prefix frontend run typecheck
# Install Chromium once:
frontend/node_modules/.bin/playwright.cmd install chromium
# Keep the real backend running in the other terminal:
$env:JIGSAW_LIVE_TEST = '1'
npm.cmd --prefix frontend run test:e2e
```

Playwright starts a production frontend on port 3000 and checks file count and
size validation, keyboard focus, a 390px mobile layout, previews, loading,
busy/retry and offline/model-unavailable states. Those error scenarios use
explicitly mocked HTTP responses. The live test uses the real trained CNN,
API and PostgreSQL: it uploads all sample pieces, receives 201, checks unique
placements and a 320px image, then reloads the persisted result without a
JavaScript page error. Without `JIGSAW_LIVE_TEST=1`, only that live test is skipped.
Screenshots and its response JSON are written under ignored `artifacts/frontend`.

This phase provides uploading and result viewing. It retains the current 5 × 5
solver constraint. Interactive piece selection/movement, saved corrections,
background jobs, confidence overlays and history browsing belong to later phases.
The app is a local development application without user accounts or access control.
