const ENDPOINT = "/evaluation/dataset";
const EXPECTED_IDS = [26, 30, 37, 40];
const ALGORITHMS = ["classical", "unet"];
const METRICS = ["dice", "iou", "sensitivity", "specificity"];

function isMetricResult(item) {
  return item && ["tp", "tn", "fp", "fn", "valid_pixels"].every((key) => Number.isInteger(item[key]) && item[key] >= 0)
    && item.metrics && METRICS.every((key) => item.metrics[key] === null
      || (Number.isFinite(item.metrics[key]) && item.metrics[key] >= 0 && item.metrics[key] <= 1));
}

export async function runDatasetEvaluation() {
  let response;
  try { response = await fetch(ENDPOINT, { method: "POST" }); }
  catch { throw new Error("Could not connect to the VisionQuest service. Check the connection and retry."); }
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error(`The evaluation service returned an unreadable response (HTTP ${response.status}).`); }
  if (!response.ok) throw new Error(typeof payload?.detail === "string" ? payload.detail : `Evaluation failed (HTTP ${response.status}).`);
  const validRows = Array.isArray(payload?.per_image) && payload.per_image.length === EXPECTED_IDS.length
    && payload.per_image.every((row, index) => row?.id === EXPECTED_IDS[index]
      && Number.isInteger(row.width) && Number.isInteger(row.height)
      && ALGORITHMS.every((name) => isMetricResult(row.algorithms?.[name])));
  const validAggregates = ALGORITHMS.every((name) => isMetricResult(payload?.algorithms?.[name]));
  if (payload?.status !== "success" || payload.aggregation !== "pooled_counts"
      || payload.subset !== "validation" || payload.ids?.join(",") !== EXPECTED_IDS.join(",")
      || typeof payload.label !== "string" || !payload.label.includes("not independent test performance")
      || !validRows || !validAggregates) {
    throw new Error("The evaluation service returned incomplete DRIVE validation results.");
  }
  return payload;
}
