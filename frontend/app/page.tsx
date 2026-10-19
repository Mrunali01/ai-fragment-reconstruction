"use client";

import { useEffect, useRef, useState } from "react";
import {
  apiResponse,
  validateFiles,
  type Health,
  type Result,
  type Scorer,
  type Search,
} from "../lib/api";

function PuzzleMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 32 32" fill="none">
      <rect x="2" y="2" width="28" height="28" rx="8" fill="currentColor" />
      <path
        d="M16 6v6c4-3 6 3 2 3h-2v11M6 16h6c-3 4 3 6 3 2v-2h11"
        stroke="white"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path
        d="M12 16V4m-4 4 4-4 4 4M4 15v5h16v-5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [scorer, setScorer] = useState<Scorer>("cnn");
  const [search, setSearch] = useState<Search>("beam");
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [imageError, setImageError] = useState(false);
  const [sample, setSample] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function checkHealth() {
    try {
      setHealth(
        await apiResponse<Health>(
          await fetch("/api/health", {
            cache: "no-store",
            signal: AbortSignal.timeout(5000),
          }),
        ),
      );
    } catch {
      setHealth(null);
    }
  }
  useEffect(() => {
    void checkHealth();
  }, []);
  useEffect(() => {
    const urls = files.map((file) => URL.createObjectURL(file));
    setPreviews(urls);
    return () => urls.forEach((url) => URL.revokeObjectURL(url));
  }, [files]);

  async function selectFiles(selected: File[], isSample = false) {
    setChecking(true);
    setError("");
    try {
      await validateFiles(selected);
      setFiles(selected);
      setSample(isSample);
      setResult(null);
      setImageError(false);
      window.history.replaceState(null, "", window.location.pathname);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not read fragments.",
      );
    } finally {
      setChecking(false);
    }
  }

  async function loadSample() {
    setChecking(true);
    setError("");
    try {
      const pieces = await Promise.all(
        Array.from({ length: 25 }, async (_, i) => {
          const name = `piece_${String(i).padStart(3, "0")}.png`;
          const response = await fetch(`/demo/${name}`);
          if (!response.ok)
            throw new Error(
              "The sample could not be loaded. Please upload your fragments.",
            );
          return new File([await response.blob()], name, { type: "image/png" });
        }),
      );
      await selectFiles(pieces, true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sample unavailable.");
    } finally {
      setChecking(false);
    }
  }

  async function solve() {
    setBusy(true);
    setError("");
    setResult(null);
    setImageError(false);
    try {
      const body = new FormData();
      files.forEach((file) => body.append("pieces", file));
      body.append("scorer", scorer);
      body.append("search", search);
      const response = await fetch("/api/reconstructions", {
        method: "POST",
        body,
      });
      const completed = await apiResponse<Result>(response);
      setResult(completed);
      // A reload can retrieve this persisted result without re-uploading fragments.
      window.history.replaceState(
        null,
        "",
        `?result=${encodeURIComponent(completed.id)}`,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Cannot reach the solver. Check the backend and try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("result");
    if (!id) return;
    if (!/^[0-9a-f-]{36}$/i.test(id)) {
      setError("Invalid saved result link.");
      return;
    }
    const controller = new AbortController();
    setBusy(true);
    fetch(`/api/reconstructions/${id}`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(apiResponse<Result>)
      .then(setResult)
      .catch((err) => {
        if (!controller.signal.aborted)
          setError(
            err instanceof Error ? err.message : "Could not load this result.",
          );
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });
    return () => controller.abort();
  }, []);

  const disabled = busy || checking;
  return (
    <div className="min-h-screen">
      <header className="site-header">
        <a href="/" className="brand">
          <PuzzleMark />
          <span>
            jigsaw<span className="brand-ai">ai</span>
          </span>
        </a>
        <span className="header-label">THE RECONSTRUCTION LAB</span>
        <div className="service-state">
          <span className={`status-dot ${health ? "online" : ""}`} />
          {health ? "Solver connected" : "Solver offline"}
          <button
            onClick={checkHealth}
            className="text-button"
            aria-label="Refresh solver status"
          >
            ↻
          </button>
        </div>
      </header>
      <main className="page-shell">
        <section className="hero">
          <div>
            <div className="eyebrow">
              <span /> A LITTLE CHAOS. A BIGGER PICTURE.
            </div>
            <h1>
              Every piece
              <br />
              belongs <em>somewhere.</em>
            </h1>
            <p>
              Give your scattered fragments a fresh perspective.
              <br className="hidden sm:block" /> Let AI bring the picture back
              together.
            </p>
          </div>
          <div className="hero-note">
            <div className="hero-grid" aria-hidden="true">
              {Array.from({ length: 9 }, (_, i) => (
                <span key={i} />
              ))}
            </div>
            <span>
              Small pieces.
              <br />
              New possibilities.
            </span>
          </div>
        </section>
        <div className="workspace-heading">
          <h2>Your workspace</h2>
          <span>
            5 × 5 puzzle <span className="dot-separator">·</span> 25 fragments
          </span>
        </div>
        <div className="workspace">
          <section
            className="panel input-panel"
            aria-labelledby="input-heading"
          >
            <div className="panel-heading">
              <div>
                <span className="step">01</span>
                <h3 id="input-heading">The pieces</h3>
              </div>
              <span className="pill">{files.length} / 25</span>
            </div>
            <fieldset disabled={disabled} className="contents">
              <input
                ref={input}
                id="fragments"
                type="file"
                multiple
                accept=".png,.jpg,.jpeg"
                className="sr-only"
                aria-label="Choose 25 fragments"
                onChange={(event) => {
                  void selectFiles(Array.from(event.target.files ?? []));
                  event.target.value = "";
                }}
              />
              <button
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onClick={() => input.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault();
                  if (!disabled) setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  if (!disabled)
                    void selectFiles(Array.from(event.dataTransfer.files));
                }}
              >
                <span className="upload-icon">
                  <UploadIcon />
                </span>
                <strong>
                  {checking
                    ? "Checking your fragments…"
                    : files.length
                      ? "Replace your fragments"
                      : "Drop your fragments here"}
                </strong>
                <span>
                  or <u>browse files</u> to get started
                </span>
                <small>25 PNG or JPEG files · 64 × 64 px · RGB</small>
              </button>
              {previews.length > 0 && (
                <div className="fragment-grid" aria-label="Uploaded fragments">
                  {previews.map((src, i) => (
                    <img
                      key={src}
                      src={src}
                      alt={`Fragment ${i + 1}: ${files[i]?.name ?? ""}`}
                      title={files[i]?.name}
                    />
                  ))}
                </div>
              )}
              {!files.length && (
                <div className="sample-prompt">
                  <span>Just looking around?</span>
                  <button className="text-button" onClick={loadSample}>
                    Try a sample puzzle <span aria-hidden="true">↗</span>
                  </button>
                </div>
              )}
              {sample && (
                <p className="attribution">
                  Sample:{" "}
                  <a
                    href="https://www.flickr.com/photos/philon/2461050066"
                    target="_blank"
                    rel="noreferrer"
                  >
                    “Scratch” by Philo Nordlund
                  </a>{" "}
                  ·{" "}
                  <a
                    href="https://creativecommons.org/licenses/by/2.0/"
                    target="_blank"
                    rel="noreferrer"
                  >
                    CC BY 2.0
                  </a>
                  . Resized, cropped and sliced.
                </p>
              )}
              <div className="settings">
                <label htmlFor="scorer">Matching method</label>
                <select
                  id="scorer"
                  value={scorer}
                  onChange={(e) => setScorer(e.target.value as Scorer)}
                >
                  <option value="cnn">Trained CNN</option>
                  <option value="classical">Classical edge matching</option>
                </select>
                <label htmlFor="search">Assembly search</label>
                <select
                  id="search"
                  value={search}
                  onChange={(e) => setSearch(e.target.value as Search)}
                >
                  <option value="beam">Beam search · recommended</option>
                  <option value="groups">Group refinement</option>
                  <option value="swaps">Swap refinement</option>
                  <option value="greedy">Greedy · fastest</option>
                </select>
              </div>
              {health && !health.cnn_available && scorer === "cnn" && (
                <p className="notice">
                  The trained CNN is unavailable. Select classical edge matching
                  or configure the model checkpoint.
                </p>
              )}
              <button
                className="solve-button"
                onClick={solve}
                disabled={
                  disabled ||
                  files.length !== 25 ||
                  (health !== null && !health.cnn_available && scorer === "cnn")
                }
              >
                {busy ? (
                  <>
                    <span className="spinner" /> Assembling your picture…
                  </>
                ) : (
                  <>
                    Reconstruct puzzle <span aria-hidden="true">↗</span>
                  </>
                )}
              </button>
            </fieldset>
            <p className="input-footnote">
              Keep pieces upright. Use every piece from one image.
            </p>
          </section>
          <section
            className="panel result-panel"
            aria-labelledby="result-heading"
            aria-busy={busy}
          >
            <div className="panel-heading">
              <div>
                <span className="step">02</span>
                <h3 id="result-heading">The bigger picture</h3>
              </div>
              <span className={`pill ${result ? "success-pill" : ""}`}>
                {busy ? "Working" : result ? "Assembled" : "Awaiting pieces"}
              </span>
            </div>
            {error && (
              <div className="error" role="alert">
                {error}
              </div>
            )}
            <div className={`result-stage ${result ? "has-result" : ""}`}>
              {result ? (
                <img
                  className="result-image"
                  src={`/api${result.image_url}`}
                  alt="Reconstructed 5 by 5 puzzle"
                  onError={() => setImageError(true)}
                />
              ) : (
                <div className="empty-state">
                  <div
                    className={`empty-grid ${busy ? "is-working" : ""}`}
                    aria-hidden="true"
                  >
                    {Array.from({ length: 25 }, (_, i) => (
                      <span key={i} style={{ animationDelay: `${i * 50}ms` }} />
                    ))}
                  </div>
                  <h4>
                    {busy
                      ? "Finding connections…"
                      : "A picture waiting to happen"}
                  </h4>
                  <p>
                    {busy
                      ? "Comparing edges and assembling all 25 pieces."
                      : "Upload your pieces, choose a method, and let the connections emerge."}
                  </p>
                </div>
              )}
            </div>
            {imageError && (
              <p className="error" role="alert">
                The saved result image could not be loaded. Check that its image
                file is still available.
              </p>
            )}
            <div aria-live="polite" className="result-caption">
              {result ? (
                <>
                  <div>
                    <strong>25 pieces. One arrangement.</strong>
                    <p>
                      {result.scorer === "cnn"
                        ? "Trained CNN"
                        : "Classical edge matching"}{" "}
                      · {result.search} search
                    </p>
                  </div>
                  <a
                    className="download-link"
                    href={`/api${result.image_url}`}
                    download
                  >
                    Save image <span aria-hidden="true">↓</span>
                  </a>
                </>
              ) : (
                <>
                  <span className="caption-icon" aria-hidden="true">
                    ✧
                  </span>
                  <p>Your reconstructed image will appear here.</p>
                </>
              )}
            </div>
            {result && (
              <p className="estimate-note">
                This arrangement is the solver’s best estimate. Repeated
                patterns and plain backgrounds can lead to incorrect matches.
              </p>
            )}
          </section>
        </div>
        <section className="process-strip" aria-label="How it works">
          <div>
            <span>01 / UPLOAD</span>
            <p>A full set of square fragments.</p>
          </div>
          <div>
            <span>02 / CONNECT</span>
            <p>AI looks for neighboring edges.</p>
          </div>
          <div>
            <span>03 / RECONSTRUCT</span>
            <p>The picture comes into focus.</p>
          </div>
        </section>
      </main>
      <footer className="site-footer">
        <span>JIGSAW AI</span>
        <p>Finding connections, one piece at a time.</p>
        <span>BUILT WITH CURIOSITY ↗</span>
      </footer>
      <p className="sample-credit">
        Sample image:{" "}
        <a href="https://www.flickr.com/photos/philon/2461050066">
          “Scratch” by Philo Nordlund
        </a>
        {" · "}
        <a href="https://creativecommons.org/licenses/by/2.0/">CC BY 2.0</a>.
        Resized, cropped and sliced into fragments.
      </p>
    </div>
  );
}
