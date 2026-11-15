const SEGMENT_ENDPOINT = "/segment";
const METHODS = ["classical", "unet", "both"];
const ALGORITHMS = ["classical", "unet"];

function describeError(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((issue) => issue.msg).filter(Boolean).join(" ") || fallback;
  }
  return fallback;
}

function validateResponse(payload, method) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("The segmentation service returned an invalid response.");
  }
  if (!METHODS.includes(payload.method) || payload.method !== method) {
    throw new Error("The segmentation service response did not match the selected method.");
  }
  if (!Number.isInteger(payload.width) || !Number.isInteger(payload.height)
      || payload.width < 1 || payload.height < 1) {
    throw new Error("The segmentation service response is missing valid image dimensions.");
  }
  if (!["success", "partial_success", "failed"].includes(payload.status)) {
    throw new Error("The segmentation service returned an unknown processing status.");
  }
  if (!payload.algorithms || typeof payload.algorithms !== "object") {
    throw new Error("The segmentation service response is missing method results.");
  }

  const expected = method === "both" ? ALGORITHMS : [method];
  for (const name of expected) {
    const outcome = payload.algorithms[name];
    if (!outcome || !["success", "failed"].includes(outcome.status)
        || !Number.isFinite(outcome.processing_time_ms)
        || outcome.processing_time_ms < 0) {
      throw new Error(`The segmentation service returned an invalid ${name} result.`);
    }
    if (outcome.status === "success"
        && (typeof outcome.mask_data_url !== "string"
          || !outcome.mask_data_url.startsWith("data:image/png;base64,"))) {
      throw new Error(`The segmentation service returned an invalid ${name} vessel map.`);
    }
    if (outcome.status === "failed" && typeof outcome.error !== "string") {
      throw new Error(`The segmentation service did not explain why ${name} failed.`);
    }
  }
  return payload;
}

export async function segmentImage(file, method) {
  const formData = new FormData();
  formData.append("file", file, file.name);
  formData.append("method", method);

  let response;
  try {
    response = await fetch(SEGMENT_ENDPOINT, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Could not connect to the VisionQuest service. Check the connection and retry.");
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`The segmentation service returned an unreadable response (HTTP ${response.status}).`);
  }

  if (!response.ok && payload?.status !== "failed") {
    throw new Error(describeError(payload, `The request failed (HTTP ${response.status}).`));
  }
  return validateResponse(payload, method);
}
