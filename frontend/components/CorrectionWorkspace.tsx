"use client";
import { useEffect, useState } from "react";
import PuzzleCanvas from "./PuzzleCanvas";
import { apiResponse, type Result } from "../lib/api";
import CompatibilityEvidence from "./CompatibilityEvidence";

type Correction = {
  id: string;
  result_id: string;
  created_at: string;
  order: number[];
  note: string;
  image_url: string;
};

export default function CorrectionWorkspace({ result }: { result: Result }) {
  const initial = result.arrangement.map((piece) => piece.upload_index);
  const [order, setOrder] = useState(initial);
  const [versions, setVersions] = useState<Correction[]>([]);
  const [version, setVersion] = useState<Correction | null>(null);
  const [note, setNote] = useState("");
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);
  const [moreVersions, setMoreVersions] = useState(false);
  useEffect(() => {
    fetch(`/api/reconstructions/${result.id}/corrections`, {
      cache: "no-store",
    })
      .then(apiResponse<Correction[]>)
      .then((rows) => {
        setVersions(rows);
        setMoreVersions(rows.length === 50);
      })
      .catch((error) => setMessage(error.message));
  }, [result.id]);
  async function save() {
    setSaving(true);
    setMessage("");
    try {
      const saved = await apiResponse<Correction>(
        await fetch(`/api/reconstructions/${result.id}/corrections`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ order, note }),
        }),
      );
      setVersions((previous) => [saved, ...previous]);
      setMessage(
        "Correction saved as a new version. The original solver result is preserved.",
      );
      setVersion(saved);
      setOrder(saved.order);
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Could not save correction.",
      );
    } finally {
      setSaving(false);
    }
  }
  return (
    <div>
      <PuzzleCanvas
        key={version?.id ?? "original"}
        result={result}
        startingOrder={version?.order}
        onChange={setOrder}
      />
      <CompatibilityEvidence result={result} />
      <section className="canvas-panel" aria-label="Saved corrections">
        <h3>Save your correction</h3>
        <p className="canvas-help">
          Human arrangements are saved separately. Saving does not verify
          reconstruction accuracy.
        </p>
        <label className="canvas-help" htmlFor="correction-note">
          Optional note
        </label>
        <input
          id="correction-note"
          maxLength={500}
          value={note}
          onChange={(event) => setNote(event.target.value)}
          className="note-input"
        />
        <div className="canvas-actions">
          <button onClick={save} disabled={saving}>
            {saving ? "Saving correction…" : "Save correction"}
          </button>
          <select
            aria-label="Load saved correction"
            value={version?.id ?? ""}
            onChange={(event) => {
              const selected =
                versions.find((row) => row.id === event.target.value) ?? null;
              setVersion(selected);
              setOrder(selected?.order ?? initial);
            }}
          >
            <option value="">Original solver arrangement</option>
            {versions.map((row, index) => (
              <option key={row.id} value={row.id}>
                Correction {versions.length - index} ·{" "}
                {new Date(row.created_at).toLocaleString()}
              </option>
            ))}
          </select>
          {version && (
            <a href={`/api${version.image_url}`} download>
              Save corrected PNG
            </a>
          )}
        </div>
        <p role="status" className="canvas-help">
          {message}
        </p>
        {moreVersions && (
          <div className="canvas-actions">
            <button
              onClick={async () => {
                try {
                  const rows = await apiResponse<Correction[]>(
                    await fetch(
                      `/api/reconstructions/${result.id}/corrections?offset=${versions.length}&limit=50`,
                      { cache: "no-store" },
                    ),
                  );
                  setVersions((previous) => [...previous, ...rows]);
                  setMoreVersions(rows.length === 50);
                } catch (error) {
                  setMessage(
                    error instanceof Error
                      ? error.message
                      : "Could not load more corrections.",
                  );
                }
              }}
            >
              More saved corrections
            </button>
          </div>
        )}
        {version && (
          <div aria-label="Original and corrected comparison">
            <p className="canvas-help">
              {version.order.filter((piece, i) => piece !== initial[i]).length}{" "}
              positions changed from the original solver arrangement. This is
              edit count, not accuracy.
            </p>
            <div className="comparison-grid">
              <figure>
                <img
                  src={`/api${result.image_url}`}
                  alt="Original solver arrangement comparison"
                />
                <figcaption>Original solver arrangement</figcaption>
              </figure>
              <figure>
                <img
                  src={`/api${version.image_url}`}
                  alt="Saved human correction comparison"
                />
                <figcaption>
                  Saved human correction
                  {version.note ? ` · ${version.note}` : ""}
                </figcaption>
              </figure>
            </div>
          </div>
        )}
        <p className="canvas-help">
          Downloads use the selected saved version. Save any new edits first.
        </p>
        <div className="canvas-actions" aria-label="Export reconstruction">
          {(["zip", "json", "csv"] as const).map((format) => (
            <a
              key={format}
              href={`/api/reconstructions/${result.id}/export?format=${format}${version ? `&correction_id=${version.id}` : ""}`}
              download
            >
              Download {format.toUpperCase()}
              {version ? " correction" : " original"}
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
