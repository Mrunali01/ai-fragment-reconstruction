import type { Result } from "../lib/api";

export default function CompatibilityEvidence({ result }: { result: Result }) {
  const evidence = result.diagnostics;
  if (!evidence)
    return (
      <p className="canvas-help">
        Matching diagnostics are unavailable for this older reconstruction.
      </p>
    );
  const weakest = [...evidence.edges]
    .sort((a, b) => a.margin - b.margin)
    .slice(0, 8);
  return (
    <section className="canvas-panel" aria-label="Compatibility evidence">
      <h3>Where the matches are uncertain</h3>
      <p className="canvas-help">
        Original solver arrangement only. {evidence.warning}
      </p>
      <div className="metric-row">
        <div>
          <strong>{evidence.edges.length}</strong>
          <span>Placed borders</span>
        </div>
        <div>
          <strong>
            {(evidence.top_ranked_border_fraction * 100).toFixed(1)}%
          </strong>
          <span>Chosen borders tied for best local score</span>
        </div>
        <div>
          <strong>{evidence.mean_border_cost.toFixed(4)}</strong>
          <span>Mean matching cost (lower is better)</span>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <caption>Eight weakest relative margins</caption>
          <thead>
            <tr>
              <th>Fragment</th>
              <th>Direction</th>
              <th>Neighbor</th>
              <th>Rank / 24</th>
              <th>Margin</th>
              <th>Best alternative</th>
            </tr>
          </thead>
          <tbody>
            {weakest.map((edge) => (
              <tr key={`${edge.piece}-${edge.direction}`}>
                <td>{edge.piece + 1}</td>
                <td>{edge.direction}</td>
                <td>{edge.neighbor + 1}</td>
                <td>{edge.rank}</td>
                <td>{edge.margin.toFixed(3)}</td>
                <td>{edge.alternatives[0].piece + 1}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details className="canvas-help">
        <summary>How these scores are calculated</summary>
        <p>{evidence.definition}</p>
        <p>
          A small margin means a close alternative. A negative margin means
          another piece scored better locally. Global assembly constraints can
          still favor this placement.
        </p>
      </details>
    </section>
  );
}
