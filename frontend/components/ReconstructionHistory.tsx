"use client";
import { useEffect, useState } from "react";
import { apiResponse, type Result } from "../lib/api";

function pieceFiles(result: Result) {
  return result.arrangement
    .map((piece) => piece.sha256)
    .sort()
    .join(":");
}

export default function ReconstructionHistory({
  refreshKey,
}: {
  refreshKey: string;
}) {
  const [items, setItems] = useState<Result[]>([]);
  const [more, setMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Result[]>([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [jobs, setJobs] = useState<
    {
      id: string;
      status: string;
      result_id: string | null;
      error: string | null;
    }[]
  >([]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setMessage("");
    fetch(`/api/history?limit=10&offset=${offset}`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(apiResponse<{ items: Result[]; has_more: boolean }>)
      .then((data) => {
        setItems(data.items);
        setMore(data.has_more);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setMessage(error.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    fetch("/api/jobs?limit=5", { signal: controller.signal, cache: "no-store" })
      .then(apiResponse<typeof jobs>)
      .then(setJobs)
      .catch(() => {});
    return () => controller.abort();
  }, [offset, refreshKey]);
  function compare(result: Result) {
    setMessage("");
    if (selected.some((row) => row.id === result.id)) {
      setSelected(selected.filter((row) => row.id !== result.id));
      return;
    }
    if (selected.length >= 2) {
      setMessage("Select at most two attempts. Uncheck an attempt first.");
      return;
    }
    if (selected.length && pieceFiles(selected[0]) !== pieceFiles(result)) {
      setMessage(
        "Choose attempts made from the same uploaded piece files to compare.",
      );
      return;
    }
    setSelected([...selected, result]);
  }
  const changed =
    selected.length === 2
      ? selected[0].arrangement.filter(
          (piece, i) => piece.sha256 !== selected[1].arrangement[i].sha256,
        ).length
      : null;
  return (
    <section className="canvas-panel" aria-label="Reconstruction history">
      <div className="panel-heading">
        <h3>Reconstruction history</h3>
        <span className="pill">Saved attempts</span>
      </div>
      <p className="canvas-help">
        Open an attempt to view its original result and saved corrections.
        Select two attempts using the same piece files to compare.
      </p>
      {message && (
        <p role="alert" className="error">
          {message}
        </p>
      )}
      {loading && (
        <p role="status" className="canvas-help">
          Loading history…
        </p>
      )}
      {!loading && !items.length && !message && (
        <p className="canvas-help">
          Your completed reconstructions will appear here.
        </p>
      )}
      <div className="history-grid">
        {items.map((row) => (
          <article key={row.id}>
            <a href={`/?result=${row.id}`}>
              <img
                src={`/api${row.image_url}`}
                alt={`Saved ${row.scorer} reconstruction`}
              />
              <strong>
                {row.scorer === "cnn" ? "Trained CNN" : "Classical"} ·{" "}
                {row.search}
              </strong>
            </a>
            <p>{new Date(row.created_at).toLocaleString()}</p>
            <label>
              <input
                type="checkbox"
                aria-label={`Compare attempt ${row.id}`}
                checked={selected.some((item) => item.id === row.id)}
                onChange={() => compare(row)}
              />{" "}
              Compare
            </label>
          </article>
        ))}
      </div>
      <div className="canvas-actions">
        <button
          disabled={loading || offset === 0}
          onClick={() => setOffset(Math.max(0, offset - 10))}
        >
          Previous attempts
        </button>
        <button
          disabled={loading || !more}
          onClick={() => setOffset(offset + 10)}
        >
          More attempts
        </button>
      </div>
      {selected.length === 2 && (
        <div aria-label="Attempt comparison">
          <h4>Side-by-side comparison</h4>
          <p className="canvas-help">
            {changed} of 25 positions contain different uploaded file content.
            This is arrangement difference, not accuracy. Pixel-identical files
            with different encodings may not be recognized as equivalent.
          </p>
          <div className="comparison-grid">
            {selected.map((row) => (
              <figure key={row.id}>
                <img
                  src={`/api${row.image_url}`}
                  alt={`${row.scorer} ${row.search} comparison`}
                />
                <figcaption>
                  {row.scorer} · {row.search}
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      )}
      {jobs.length > 0 && (
        <details className="canvas-help">
          <summary>Recent background jobs</summary>
          <ul>
            {jobs.map((job) => (
              <li key={job.id}>
                <a
                  href={
                    job.result_id
                      ? `/?result=${job.result_id}`
                      : `/?job=${job.id}`
                  }
                >
                  {job.status} · {job.id.slice(0, 8)}
                </a>
                {job.error ? ` — ${job.error}` : ""}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
