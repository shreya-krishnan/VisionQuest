const names = { classical: "Classical", unet: "U-Net" };
const initialPreviewState = new WeakMap();

function setImageSource(image, source, visible = true) {
  if (!image) return;
  if (source) image.src = source;
  image.hidden = !(visible && source);
  image.style.display = visible && source ? "" : "none";
}

function formatTime(milliseconds) {
  return `${milliseconds < 100 ? milliseconds.toFixed(1) : Math.round(milliseconds)} ms`;
}

function renderResultCard(root, name, state, originalUrl, originalAvailable) {
  const card = root.querySelector(`[data-result="${name}"]`);
  if (!card) return;
  const caption = card.querySelector("[data-result-status]");
  const empty = card.querySelector(".empty");
  const emptyMessage = empty?.querySelector("span");
  const thumb = card.querySelector(".thumb");
  const originalLayer = thumb?.querySelector("[data-original-layer]");
  const maskLayer = thumb?.querySelector("[data-mask-layer]");
  const tabs = thumb?.querySelector(".tabs");
  const outcome = state.result?.algorithms?.[name];
  const selected = state.method === "both" || state.method === name;
  card.hidden = !selected;

  if (state.processing && selected) {
    thumb.dataset.mode = "mask";
    thumb.dataset.outputState = "processing";
    if (caption) caption.textContent = "Processing…";
    if (emptyMessage) emptyMessage.textContent = "Waiting for the vessel map…";
    if (empty) empty.hidden = false;
    setImageSource(originalLayer, null, false);
    setImageSource(maskLayer, null, false);
    if (tabs) tabs.hidden = true;
    return;
  }

  if (!state.result) {
    thumb.dataset.mode = "mask";
    thumb.dataset.outputState = "pending";
    if (caption) caption.textContent = "Not run";
    if (emptyMessage) emptyMessage.textContent = name === "classical"
      ? "Run Classical processing to generate a vessel map."
      : "Run U-Net to generate a vessel map.";
    if (empty) empty.hidden = false;
    setImageSource(originalLayer, null, false);
    setImageSource(maskLayer, null, false);
    if (tabs) tabs.hidden = true;
    return;
  }

  if (!outcome) {
    thumb.dataset.outputState = "failed";
    if (caption) caption.textContent = "Not selected";
    if (emptyMessage) emptyMessage.textContent = "This method was not selected for this run.";
    if (empty) empty.hidden = false;
    setImageSource(originalLayer, null, false);
    setImageSource(maskLayer, null, false);
    if (tabs) tabs.hidden = true;
    return;
  }

  if (outcome.status === "success") {
    thumb.dataset.outputState = "success";
    if (caption) caption.textContent = `Success · ${formatTime(outcome.processing_time_ms)}`;
    if (empty) empty.hidden = true;
    if (maskLayer) maskLayer.alt = `${names[name]} predicted vessel mask`;
    setImageSource(originalLayer, originalUrl, originalAvailable);
    setImageSource(maskLayer, outcome.mask_data_url, true);
    if (tabs) {
      tabs.hidden = false;
      const overlayTab = tabs.querySelector('[data-mode="overlay"]');
      if (overlayTab) {
        overlayTab.disabled = !originalAvailable;
        overlayTab.title = originalAvailable ? "Show the vessel map over the uploaded image" : "Original image preview is unavailable";
      }
    }
  } else {
    thumb.dataset.outputState = "failed";
    const duration = Number.isFinite(outcome.processing_time_ms)
      ? ` · ${formatTime(outcome.processing_time_ms)}`
      : "";
    if (caption) caption.textContent = `Failed${duration}`;
    if (emptyMessage) emptyMessage.textContent = outcome.error || `${names[name]} processing failed.`;
    if (empty) empty.hidden = false;
    setImageSource(originalLayer, null, false);
    setImageSource(maskLayer, null, false);
    if (tabs) tabs.hidden = true;
  }
}

export function renderVisionQuest(root, state) {
  const status = root.querySelector("#status");
  const runButton = root.querySelector("#runBtn");
  const resetSessionButton = root.querySelector("#resetSessionBtn");
  const showResultsButton = root.querySelector("#showResultsBtn");
  const showResultsHint = root.querySelector("#showResultsHint");
  const canvas = root.querySelector("#canvas");
  const fileName = root.querySelector("#fileName");
  const uploadHint = root.querySelector("#uploadHint");
  const fileDimensions = root.querySelector("#fileDimensions");
  const uploadInput = root.querySelector("#uploadInput");
  const inputPreview = root.querySelector("#inputPreview");
  const previewMask = root.querySelector("#previewMask");
  const maskLabel = root.querySelector("#previewMaskLabel");
  const previewDisclaimer = root.querySelector("#previewDisclaimer");
  const compareRange = root.querySelector("#ba input[type='range']");
  if (inputPreview && !initialPreviewState.has(inputPreview)) {
    initialPreviewState.set(inputPreview, {
      image: inputPreview.getAttribute("src"),
      mask: previewMask?.getAttribute("src"),
      maskLabel: maskLabel?.textContent,
      disclaimer: previewDisclaimer?.textContent,
    });
  }
  const viewerOriginal = root.querySelector("#viewerOriginal");
  const originalEmpty = root.querySelector(".original-empty");
  const originalImageStatus = root.querySelector("#originalImageStatus");
  const resultsLayout = root.querySelector("#results .rs-layout");
  const selectedMethodLabel = root.querySelector("#selectedMethodLabel");
  if (selectedMethodLabel) {
    const methodLabel = state.method === "unet" ? "U-Net" : state.method === "both" ? "Both methods" : "Classical";
    selectedMethodLabel.textContent = `Selected method: ${methodLabel}`;
  }
  if (showResultsButton) {
    const canShowResults = !state.processing
      && ["success", "partial_success"].includes(state.result?.status);
    showResultsButton.disabled = !canShowResults;
    showResultsButton.setAttribute("aria-disabled", canShowResults ? "false" : "true");
    if (showResultsHint) {
      showResultsHint.textContent = canShowResults
        ? "The latest vessel maps are ready to view."
        : state.processing ? "Results will be ready when processing finishes."
          : "Complete a segmentation run to view its results.";
    }
  }
  if (resultsLayout) {
    resultsLayout.dataset.panelCount = state.method === "both" ? "3" : "2";
  }

  if (status) {
    status.textContent = state.statusMessage;
    status.dataset.state = state.statusKind;
    status.title = state.statusMessage;
  }
  if (runButton) {
    runButton.disabled = state.processing;
    runButton.setAttribute("aria-disabled", state.processing ? "true" : "false");
    runButton.setAttribute("aria-busy", state.processing ? "true" : "false");
  }
  if (resetSessionButton) resetSessionButton.disabled = state.processing;
  if (uploadInput) uploadInput.disabled = state.processing;
  root.querySelectorAll("input[name='model']").forEach((input) => {
    input.disabled = state.processing;
    input.checked = input.value === state.method;
  });
  if (canvas) canvas.classList.toggle("scanning", state.processing);
  if (fileName) fileName.textContent = state.selectedFile?.name || "Upload an image";
  if (uploadHint && state.selectedFile) {
    uploadHint.textContent = state.previewAvailable
      ? "Image selected · choose another file to replace it"
      : "Image selected · preview unavailable in this browser";
  }
  if (uploadHint && !state.selectedFile) uploadHint.textContent = "PNG, JPG or TIFF";
  if (fileDimensions) {
    fileDimensions.textContent = state.imageDimensions
      ? `${state.imageDimensions.width} × ${state.imageDimensions.height} px`
      : state.selectedFile ? "Dimensions unavailable for this preview" : "Dimensions appear after selection";
  }

  const hasSelectedFile = Boolean(state.selectedFile);
  const showPreview = hasSelectedFile && state.previewAvailable && Boolean(state.previewUrl);
  setImageSource(viewerOriginal, showPreview ? state.previewUrl : null, showPreview);
  if (originalEmpty) {
    originalEmpty.hidden = showPreview;
    if (hasSelectedFile && !showPreview) {
      originalEmpty.querySelector("span").textContent = "This image preview is unavailable in this browser.";
    } else if (!hasSelectedFile) {
      originalEmpty.querySelector("span").textContent = "Upload an image in Workspace to view it here.";
    }
  }
  if (originalImageStatus) {
    originalImageStatus.textContent = state.selectedFile
      ? `${state.selectedFile.name}${state.imageDimensions ? ` · ${state.imageDimensions.width} × ${state.imageDimensions.height}` : ""}`
      : "Awaiting upload";
  }
  if (hasSelectedFile) {
    setImageSource(inputPreview, showPreview ? state.previewUrl : null, showPreview);
  } else if (inputPreview) {
    const initial = initialPreviewState.get(inputPreview);
    setImageSource(inputPreview, initial?.image, true);
    if (previewMask) setImageSource(previewMask, initial?.mask, true);
    if (maskLabel) maskLabel.textContent = initial?.maskLabel || "Illustrative vessel map";
    if (previewDisclaimer) previewDisclaimer.textContent = initial?.disclaimer || "Illustration only · not a model prediction";
    if (compareRange) compareRange.value = "50";
    const beforeAfter = root.querySelector("#ba");
    if (beforeAfter) beforeAfter.style.setProperty("--pos", "50%");
  }
  if (previewMask && showPreview && !state.result) {
    setImageSource(previewMask, state.previewUrl);
    if (maskLabel) maskLabel.textContent = "Input image preview";
    if (previewDisclaimer) previewDisclaimer.textContent = "Input image · no segmentation output";
  } else if (previewMask && hasSelectedFile && !showPreview) {
    setImageSource(previewMask, null, false);
    if (maskLabel) maskLabel.textContent = "Preview unavailable";
    if (previewDisclaimer) previewDisclaimer.textContent = "No image preview available";
  }

  const successful = ["classical", "unet"]
    .map((name) => ({ name, outcome: state.result?.algorithms?.[name] }))
    .find(({ outcome }) => outcome?.status === "success");
  if (successful) {
    setImageSource(previewMask, successful.outcome.mask_data_url);
    if (maskLabel) maskLabel.textContent = `${names[successful.name]} vessel map`;
    if (previewDisclaimer) previewDisclaimer.textContent = "Segmentation output · not a clinical diagnosis";
  }

  renderResultCard(root, "classical", state, state.previewUrl, showPreview);
  renderResultCard(root, "unet", state, state.previewUrl, showPreview);
  root.dispatchEvent(new CustomEvent("visionquest:results-updated"));
}
