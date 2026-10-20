"use client";

import { useRef, useState } from "react";
import type { Result } from "../lib/api";
import ReconstructAgain from "./ReconstructAgain";

export default function PuzzleCanvas({
  result,
  onChange,
  startingOrder,
}: {
  result: Result;
  onChange?: (order: number[]) => void;
  startingOrder?: number[];
}) {
  const initial = result.arrangement.map((piece) => piece.upload_index);
  const [order, setOrder] = useState(startingOrder ?? initial);
  const [selected, setSelected] = useState<number | null>(null);
  const [undo, setUndo] = useState<number[][]>([]);
  const [heatmap, setHeatmap] = useState(false);
  const [message, setMessage] = useState(
    "Select two pieces to swap them, or drag a piece onto another.",
  );
  const pointer = useRef<{ position: number; x: number; y: number } | null>(
    null,
  );
  const suppressClick = useRef(false);

  function change(next: number[]) {
    setUndo((previous) => [...previous.slice(-49), order]);
    setOrder(next);
    setSelected(null);
    onChange?.(next);
  }
  function swap(a: number, b: number) {
    if (a === b) return;
    const next = [...order];
    [next[a], next[b]] = [next[b], next[a]];
    change(next);
    setMessage(`Swapped positions ${a + 1} and ${b + 1}.`);
  }
  function choose(position: number) {
    if (suppressClick.current) {
      suppressClick.current = false;
      return;
    }
    if (selected === null) {
      setSelected(position);
      setMessage(
        `Position ${position + 1} selected. Choose another piece to swap.`,
      );
    } else {
      swap(selected, position);
      setSelected(null);
    }
  }
  return (
    <section className="canvas-panel" aria-label="Interactive puzzle">
      <div className="panel-heading">
        <h3>Arrange the pieces</h3>
        <span className="pill">25 unique pieces</span>
      </div>
      <p className="canvas-help">
        Move pieces to correct matches the solver got wrong. Click or tap two
        pieces to swap. You can also drag, or use Tab and Enter. Escape clears
        selection.
      </p>
      {result.diagnostics && (
        <label className="canvas-help">
          <input
            type="checkbox"
            checked={heatmap}
            onChange={(event) => setHeatmap(event.target.checked)}
          />{" "}
          Show relative edge evidence (original arrangement only: green =
          stronger margin, orange = ambiguous)
        </label>
      )}
      <div
        className="puzzle-canvas"
        role="group"
        aria-label="Editable 5 by 5 puzzle"
        onKeyDown={(event) => {
          if (event.key === "Escape") setSelected(null);
        }}
      >
        {order.map((piece, position) => {
          const source = initial.indexOf(piece);
          const edges =
            result.diagnostics?.edges.filter(
              (edge) => edge.piece === piece || edge.neighbor === piece,
            ) ?? [];
          const margin = edges.length
            ? Math.min(...edges.map((edge) => edge.margin))
            : 0;
          const showHeatmap =
            heatmap && order.every((value, i) => value === initial[i]);
          return (
            <button
              key={position}
              type="button"
              data-position={position}
              data-piece={piece}
              aria-label={`Position ${position + 1}, fragment ${piece + 1}`}
              aria-pressed={selected === position}
              onClick={() => choose(position)}
              onPointerDown={(event) => {
                pointer.current = {
                  position,
                  x: event.clientX,
                  y: event.clientY,
                };
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onPointerCancel={() => {
                pointer.current = null;
              }}
              onPointerUp={(event) => {
                const from = pointer.current;
                pointer.current = null;
                if (
                  !from ||
                  Math.hypot(event.clientX - from.x, event.clientY - from.y) < 8
                )
                  return;
                suppressClick.current = true;
                const target = document
                  .elementFromPoint(event.clientX, event.clientY)
                  ?.closest<HTMLButtonElement>("button[data-position]");
                if (
                  target &&
                  event.currentTarget.parentElement?.contains(target)
                )
                  swap(from.position, Number(target.dataset.position));
              }}
              style={{
                backgroundImage: `url(/api${result.image_url})`,
                backgroundPosition: `${(source % 5) * 25}% ${Math.floor(source / 5) * 25}%`,
              }}
            >
              {showHeatmap && (
                <i
                  aria-hidden="true"
                  className="evidence-tint"
                  style={{ background: margin > 0.05 ? "#2cbd5c" : "#ed9b35" }}
                />
              )}
              <span>{piece + 1}</span>
            </button>
          );
        })}
      </div>
      <div className="canvas-actions">
        <button
          disabled={!undo.length}
          onClick={() => {
            const previous = undo[undo.length - 1];
            setUndo(undo.slice(0, -1));
            setOrder(previous);
            setSelected(null);
            onChange?.(previous);
            setMessage("Last change undone.");
          }}
        >
          Undo move
        </button>
        <button
          onClick={() => {
            change(initial);
            setMessage("Restored the solver arrangement.");
          }}
        >
          Restore solver result
        </button>
        <ReconstructAgain result={result} order={order} />
      </div>
      <p className="canvas-help" role="status">
        {message}
      </p>
    </section>
  );
}
