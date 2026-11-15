import { useCallback, useEffect, useRef, useState } from "react";
import { segmentImage } from "../services/segmentationApi.js";
import {
  clearWorkspaceSession,
  readWorkspaceSession,
  saveWorkspaceSession,
} from "../services/workspaceSessionStore.js";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 20_000_000;
const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/tiff"]);
const EXTENSION_TYPES = new Set([".jpg", ".jpeg", ".png", ".tif", ".tiff"]);

function isAcceptedImage(file) {
  const extension = `.${file.name.split(".").pop()}`.toLowerCase();
  return ACCEPTED_TYPES.has(file.type.toLowerCase()) || EXTENSION_TYPES.has(extension);
}

function decodePreview(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
  image.onerror = () => reject(new Error("This image cannot be previewed in this browser."));
    image.src = url;
  });
}

async function createTiffPreview(file) {
  const imported = await import("utif");
  const UTIF = imported.default || imported;
  const buffer = await file.arrayBuffer();
  const directories = UTIF.decode(buffer);
  const firstImage = directories[0];
  if (!firstImage) throw new Error("This TIFF file contains no readable image.");
  const width = firstImage.t256?.[0] ?? firstImage.width;
  const height = firstImage.t257?.[0] ?? firstImage.height;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error("This TIFF image has invalid dimensions.");
  }
  if (width * height > MAX_IMAGE_PIXELS) {
    throw new RangeError("Image exceeds the 20 million pixel limit.");
  }

  UTIF.decodeImage(buffer, firstImage, directories);
  const rgba = UTIF.toRGBA8(firstImage);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not prepare a preview of this TIFF image.");
  context.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
  const png = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!png) throw new Error("Could not create a preview of this TIFF image.");
  return { width, height, url: URL.createObjectURL(png) };
}

function selectedMethods(method) {
  return method === "both" ? ["classical", "unet"] : [method];
}

async function preparePreview(file) {
  const extension = `.${file.name.split(".").pop()}`.toLowerCase();
  const isTiff = extension === ".tif" || extension === ".tiff" || file.type === "image/tiff";
  if (isTiff) return { ...(await createTiffPreview(file)), available: true };
  const url = URL.createObjectURL(file);
  try {
    const dimensions = await decodePreview(url);
    return { ...dimensions, url, available: true };
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
}

function isRestorableResult(result) {
  if (!result || !["success", "partial_success", "failed"].includes(result.status)
      || !result.algorithms || typeof result.algorithms !== "object") return false;
  return Object.values(result.algorithms).every((outcome) => (
    outcome && ["success", "failed"].includes(outcome.status)
    && (outcome.status !== "success"
      || (typeof outcome.mask_data_url === "string"
        && outcome.mask_data_url.startsWith("data:image/png;base64,")))
  ));
}

export default function useVisionQuest() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewAvailable, setPreviewAvailable] = useState(false);
  const [imageDimensions, setImageDimensions] = useState(null);
  const [method, setMethod] = useState("unet");
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [statusMessage, setStatusMessage] = useState("Drag to compare the illustrative images");
  const [statusKind, setStatusKind] = useState("idle");
  const [persistenceReady, setPersistenceReady] = useState(false);
  const selectionId = useRef(0);
  const persistenceId = useRef(0);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => {
    let mounted = true;
    const initialSelection = selectionId.current;
    async function restore() {
      try {
        const saved = await readWorkspaceSession();
        if (!saved) return;
        const validMethod = ["classical", "unet", "both"].includes(saved.method);
        const validBlob = saved.imageBlob instanceof Blob
          && saved.imageBlob.size > 0 && saved.imageBlob.size <= MAX_UPLOAD_BYTES;
        const validMetadata = typeof saved.fileName === "string"
          && saved.fileName.length > 0 && saved.fileName.length <= 255
          && typeof saved.fileType === "string"
          && Number.isInteger(saved.width) && Number.isInteger(saved.height)
          && saved.width > 0 && saved.height > 0
          && saved.width * saved.height <= MAX_IMAGE_PIXELS;
        if (!validMethod || !validBlob || !validMetadata
            || (saved.result !== null && !isRestorableResult(saved.result))) {
          await clearWorkspaceSession();
          throw new Error("The saved workspace data was incomplete or invalid.");
        }
        const file = new File([saved.imageBlob], saved.fileName, {
          type: saved.fileType,
          lastModified: Number.isFinite(saved.lastModified) ? saved.lastModified : Date.now(),
        });
        let nextPreview = null;
        let previewAvailable = false;
        try {
          nextPreview = await preparePreview(file);
          previewAvailable = true;
        } catch (error) {
          if (nextPreview?.url) URL.revokeObjectURL(nextPreview.url);
          nextPreview = null;
          const extension = `.${file.name.split(".").pop()}`.toLowerCase();
          const isTiff = extension === ".tif" || extension === ".tiff" || file.type === "image/tiff";
          if (!isTiff) throw error;
        }
        if (nextPreview && (nextPreview.width !== saved.width || nextPreview.height !== saved.height)) {
          URL.revokeObjectURL(nextPreview.url);
          await clearWorkspaceSession();
          throw new Error("The saved image dimensions did not match its preview.");
        }
        if (!mounted || selectionId.current !== initialSelection) {
          if (nextPreview?.url) URL.revokeObjectURL(nextPreview.url);
          return;
        }
        setSelectedFile(file);
        setPreviewUrl(nextPreview?.url || null);
        setPreviewAvailable(previewAvailable);
        setImageDimensions({ width: saved.width, height: saved.height });
        setMethod(saved.method);
        setResult(saved.result);
        setStatusMessage(previewAvailable
          ? `Restored ${file.name} and its latest results from this browser.`
          : `Restored ${file.name}; its TIFF preview is unavailable in this browser.`);
        setStatusKind(previewAvailable ? "ready" : "notice");
      } catch (error) {
        if (mounted && selectionId.current === initialSelection) {
          const couldNotRestore = error instanceof Error
            && !error.message.includes("unavailable")
            && !error.message.includes("blocked")
            && !error.message.includes("open workspace storage");
          if (couldNotRestore) await clearWorkspaceSession().catch(() => {});
          setStatusMessage(couldNotRestore
            ? "The saved workspace session could not be read. Start a new session and upload the image again."
            : "Browser storage is unavailable; this session may not return after a refresh.");
          setStatusKind("notice");
        }
      } finally {
        if (mounted) setPersistenceReady(true);
      }
    }
    restore();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    if (!persistenceReady || !selectedFile) return undefined;
    const currentPersistence = ++persistenceId.current;
    const operation = saveWorkspaceSession({
        imageBlob: selectedFile.slice(0, selectedFile.size, selectedFile.type),
        fileName: selectedFile.name,
        fileType: selectedFile.type,
        lastModified: selectedFile.lastModified,
        width: imageDimensions?.width,
        height: imageDimensions?.height,
        method,
        result,
      });
    operation.catch(() => {
      if (currentPersistence === persistenceId.current) {
        setStatusMessage("Could not save this session in browser storage. It may not return after a refresh.");
        setStatusKind("notice");
      }
    });
    return undefined;
  }, [persistenceReady, selectedFile, imageDimensions, method, result]);

  const chooseFile = useCallback(async (file) => {
    if (!file) return;
    const currentSelection = ++selectionId.current;
    if (!isAcceptedImage(file)) {
      setStatusMessage("Choose a PNG, JPEG, or TIFF image file.");
      setStatusKind("error");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setStatusMessage("Image exceeds the 10 MB upload limit.");
      setStatusKind("error");
      return;
    }

    let nextPreviewUrl = null;
    let canPreview = true;
    let nextDimensions = null;
    try {
      const dimensions = await preparePreview(file);
      nextPreviewUrl = dimensions.url;
      nextDimensions = { width: dimensions.width, height: dimensions.height };
      if (dimensions.width * dimensions.height > MAX_IMAGE_PIXELS) {
        if (nextPreviewUrl) URL.revokeObjectURL(nextPreviewUrl);
        setStatusMessage("Image exceeds the 20 million pixel limit.");
        setStatusKind("error");
        return;
      }
    } catch (error) {
      if (error instanceof RangeError) {
        if (nextPreviewUrl) URL.revokeObjectURL(nextPreviewUrl);
        setStatusMessage(error.message);
        setStatusKind("error");
        return;
      }
      const extension = `.${file.name.split(".").pop()}`.toLowerCase();
      const isTiff = extension === ".tif" || extension === ".tiff" || file.type === "image/tiff";
      if (!isTiff) {
        if (nextPreviewUrl) URL.revokeObjectURL(nextPreviewUrl);
        setStatusMessage("The selected file could not be previewed. Choose a valid image.");
        setStatusKind("error");
        return;
      }
      // Keep TIFF uploads usable if a browser decoder cannot parse that TIFF variant.
      canPreview = false;
    }

    if (currentSelection !== selectionId.current) {
      if (nextPreviewUrl) URL.revokeObjectURL(nextPreviewUrl);
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(nextPreviewUrl);
    setPreviewAvailable(canPreview);
    setImageDimensions(nextDimensions);
    setResult(null);
    setStatusMessage(canPreview
      ? `${file.name} is ready for segmentation.`
      : `${file.name} is ready. Its TIFF preview is unavailable in this browser.`);
    setStatusKind(canPreview ? "ready" : "notice");
  }, []);

  const changeMethod = useCallback((nextMethod) => {
    if (!["classical", "unet", "both"].includes(nextMethod)) return;
    setMethod(nextMethod);
    setResult(null);
    setStatusMessage(selectedFile ? `${selectedFile.name} is ready for segmentation.` : "Choose an image to begin.");
    setStatusKind(selectedFile ? "ready" : "idle");
  }, [selectedFile]);

  const run = useCallback(async (requestedMethod = method) => {
    if (processing) return;
    if (!selectedFile) {
      setStatusMessage("Choose an image before running segmentation.");
      setStatusKind("error");
      return;
    }

    setProcessing(true);
    setResult(null);
    setStatusMessage(`Processing with ${requestedMethod === "both" ? "both methods" : requestedMethod === "unet" ? "U-Net" : "Classical processing"}…`);
    setStatusKind("processing");
    try {
      const response = await segmentImage(selectedFile, requestedMethod);
      setResult(response);
      if (response.status === "success") {
        setStatusMessage("Segmentation complete.");
        setStatusKind("success");
      } else if (response.status === "partial_success") {
        setStatusMessage("Segmentation partially completed. See each method’s results.");
        setStatusKind("notice");
      } else {
        setStatusMessage("Segmentation failed. See the method results for details.");
        setStatusKind("error");
      }
      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Segmentation could not be completed.";
      const outcomes = Object.fromEntries(selectedMethods(requestedMethod).map((name) => [name, {
        status: "failed",
        error: message,
        processing_time_ms: null,
      }]));
      setResult({ status: "failed", method: requestedMethod, algorithms: outcomes });
      setStatusMessage(message);
      setStatusKind("error");
      return { status: "failed" };
    } finally {
      setProcessing(false);
    }
  }, [method, processing, selectedFile]);

  const resetSession = useCallback(async () => {
    selectionId.current += 1;
    setSelectedFile(null);
    setPreviewUrl(null);
    setPreviewAvailable(false);
    setImageDimensions(null);
    setMethod("unet");
    setProcessing(false);
    setResult(null);
    setStatusMessage("New session ready. Upload an image to begin.");
    setStatusKind("idle");
    const input = document.querySelector("#uploadInput");
    if (input) input.value = "";
    try {
      await clearWorkspaceSession();
    } catch {
      setStatusMessage("Session reset in this tab, but browser storage could not be cleared. Refresh may restore the previous session.");
      setStatusKind("notice");
    }
  }, []);

  return {
    selectedFile,
    previewUrl,
    previewAvailable,
    imageDimensions,
    method,
    processing,
    result,
    statusMessage,
    statusKind,
    resetSession,
    chooseFile,
    changeMethod,
    run,
  };
}
