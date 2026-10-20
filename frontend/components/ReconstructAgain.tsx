"use client";
import { useState } from "react";
import { apiResponse, type Result, type Scorer, type Search } from "../lib/api";

export default function ReconstructAgain({
  result,
  order,
}: {
  result: Result;
  order: number[];
}) {
  const [scorer, setScorer] = useState<Scorer>(result.scorer);
  const [search, setSearch] = useState<Search>(result.search);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function reconstruct() {
    setBusy(true);
    setError("");
    try {
      const job = await apiResponse<{ id: string }>(
        await fetch(`/api/reconstructions/${result.id}/reconstruct`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ order, scorer, search }),
        }),
      );
      window.location.assign(`/?job=${encodeURIComponent(job.id)}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not start reconstruction.",
      );
      setBusy(false);
    }
  }
  return (
    <section className="rerun-control" aria-label="Reconstruct current pieces">
      <fieldset disabled={busy} className="contents">
        <button className="reconstruct-button" onClick={reconstruct}>
          {busy ? "Starting reconstruction…" : "Reconstruct again"}
        </button>
        <details className="rerun-options">
          <summary>Solver options</summary>
          <div className="settings">
            <label htmlFor="rerun-scorer">Reconstruction method</label>
            <select
              id="rerun-scorer"
              value={scorer}
              onChange={(event) => setScorer(event.target.value as Scorer)}
            >
              <option value="cnn">Trained CNN</option>
              <option value="classical">Classical edge matching</option>
            </select>
            <label htmlFor="rerun-search">Reconstruction search</label>
            <select
              id="rerun-search"
              value={search}
              onChange={(event) => setSearch(event.target.value as Search)}
            >
              <option value="beam">Beam search</option>
              <option value="groups">Group refinement</option>
              <option value="swaps">Swap refinement</option>
              <option value="greedy">Greedy</option>
            </select>
          </div>
        </details>
      </fieldset>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <p className="canvas-help rerun-note">
        Reconstruct again solves these pieces afresh without another upload.
        Save a correction first to keep your manual layout. The same method may
        return the same arrangement.
      </p>
    </section>
  );
}
