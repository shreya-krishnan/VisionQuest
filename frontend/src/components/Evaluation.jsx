import { useState } from "react";
import { runDatasetEvaluation } from "../services/evaluationApi.js";

const disclaimer = "DRIVE validation subset — 4 images (IDs 26, 30, 37, 40). The U-Net checkpoint was selected using validation Dice; these results are validation performance, not independent test performance.";
const metricNames = ["dice", "iou", "sensitivity", "specificity"];
const formatMetric = (value) => value == null ? "N/A" : `${(value * 100).toFixed(2)}%`;

export default function Evaluation() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  async function evaluate() {
    if (loading) return;
    setLoading(true); setError(""); setResult(null);
    try { setResult(await runDatasetEvaluation()); }
    catch (failure) { setResult(null); setError(failure instanceof Error ? failure.message : "The DRIVE validation evaluation failed."); }
    finally { setLoading(false); }
  }
  return (
    <section id="evaluation" className="chapter ev" data-nav-bg="rgba(248,243,232,.88)" data-nav-solid="#F8F3E8" data-nav-ink="#2A1410" data-wipe="#EDE3CF" aria-labelledby="ev-title">
      <div className="wrap">
        <div className="chapter-tag"><span className="eyebrow">CH. 04 / EVALUATION</span><span className="num">DRIVE validation</span></div>
        <div className="ev-head grid12"><h2 id="ev-title" className="display">Beyond what <em>meets the eye.</em></h2><p className="lede">Measure segmentation performance against expert annotations.</p></div>
        <div className="evaluation-dataset-card"><div><h3>Fixed validation subset</h3><p>Four DRIVE images: IDs 26, 30, 37 and 40. Metrics are pooled across valid FOV pixels. No saved predictions or prior CSV scores are used.</p></div><button className="btn btn-coral" type="button" onClick={evaluate} disabled={loading} aria-busy={loading}>{loading ? "Evaluating…" : result ? "Run again" : error ? "Retry evaluation" : "Run evaluation"}</button></div>
        <p className="evaluation-disclaimer">{disclaimer}</p>
        {loading && <p className="evaluation-status" role="status">Evaluating the four DRIVE validation images with both methods…</p>}
        {error && <p className="evaluation-error" role="alert">{error}</p>}
        {result && <div className="evaluation-results" aria-live="polite">
          <h3>Pooled metrics</h3><p className="evaluation-note">Counts are pooled over the four images before calculating metrics. A metric with a zero denominator is shown as N/A.</p>
          <div className="table evaluation-table-wrap"><table><caption className="sr-only">Pooled DRIVE validation metrics for Classical and U-Net</caption><thead><tr><th scope="col">Approach</th><th scope="col">Dice / F1</th><th scope="col">IoU</th><th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">TP / TN / FP / FN</th></tr></thead><tbody>{["classical", "unet"].map((name) => { const item = result.algorithms[name]; return <tr key={name}><th scope="row">{name === "unet" ? "U-Net" : "Classical"}</th>{metricNames.map((metric) => <td key={metric}>{formatMetric(item.metrics[metric])}</td>)}<td>{item.tp} / {item.tn} / {item.fp} / {item.fn}</td></tr>; })}</tbody></table></div>
          <h3>Per-image results</h3><div className="table evaluation-table-wrap"><table><caption className="sr-only">Per-image DRIVE validation metrics</caption><thead><tr><th scope="col">ID</th><th scope="col">Approach</th><th scope="col">Dice</th><th scope="col">IoU</th><th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">TP / TN / FP / FN</th></tr></thead><tbody>{result.per_image.flatMap((image) => ["classical", "unet"].map((name) => { const item = image.algorithms[name]; return <tr key={`${image.id}-${name}`}><th scope="row">{image.id}</th><td>{name === "unet" ? "U-Net" : "Classical"}</td>{metricNames.map((metric) => <td key={metric}>{formatMetric(item.metrics[metric])}</td>)}<td>{item.tp} / {item.tn} / {item.fp} / {item.fn}</td></tr>; }))}</tbody></table></div>
          <details className="evaluation-methodology"><summary>Metric and threshold details</summary><p>{result.methodology.prediction_threshold}. {result.methodology.annotation_threshold}. {result.methodology.fov_threshold}. {result.methodology.zero_denominator}. {result.methodology.scope}</p></details>
        </div>}
        <div className="evaluation-actions"><a className="btn btn-ghost" href="#home" data-target="home">Home</a><a className="btn btn-coral" href="#results" data-target="results">Back to Results <span className="arrow" aria-hidden="true">→</span></a></div>
        <footer className="evaluation-footer"><span><b>VisionQuest</b><small>Retinal vessel segmentation</small></span><nav aria-label="Evaluation footer navigation"><a href="#home" data-target="home">Home</a><a href="#method" data-target="method">Read the Method</a></nav></footer>
      </div>
    </section>
  );
}
