"""Adapter around Poorvi's trained single-image U-Net inference package."""

from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np
from PIL import Image

from unet_inference import inference as poorvi_inference


class UnsupportedImageDimensionsError(ValueError):
    """Kept for compatibility with callers handling prior dimension failures."""


@dataclass(frozen=True)
class ImageTransform:
    """Geometry needed to restore a model mask to its original image size."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_left: int = 0
    pad_top: int = 0
    pad_right: int = 0
    pad_bottom: int = 0


def _prepare_model_input(image: Image.Image) -> tuple[np.ndarray, ImageTransform]:
    """Fit RGB image inside the model canvas without changing its aspect ratio."""
    expected_width = poorvi_inference.EXPECTED_WIDTH
    expected_height = poorvi_inference.EXPECTED_HEIGHT
    original_width, original_height = image.size

    # Keep the trained-size inference input exactly as it was before adaptation.
    if image.size == (expected_width, expected_height):
        return np.asarray(image), ImageTransform(
            original_width=original_width,
            original_height=original_height,
            resized_width=original_width,
            resized_height=original_height,
        )

    scale = min(expected_width / original_width, expected_height / original_height)
    resized_width = min(expected_width, max(1, round(original_width * scale)))
    resized_height = min(expected_height, max(1, round(original_height * scale)))
    pad_left = (expected_width - resized_width) // 2
    pad_top = (expected_height - resized_height) // 2
    pad_right = expected_width - resized_width - pad_left
    pad_bottom = expected_height - resized_height - pad_top

    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (expected_width, expected_height), color=(0, 0, 0))
    canvas.paste(resized, (pad_left, pad_top))

    return np.asarray(canvas), ImageTransform(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def _restore_mask(binary_mask: np.ndarray, transform: ImageTransform) -> np.ndarray:
    """Crop model padding, then restore the binary mask to original dimensions."""
    y_end = binary_mask.shape[0] - transform.pad_bottom
    x_end = binary_mask.shape[1] - transform.pad_right
    cropped = binary_mask[
        transform.pad_top:y_end,
        transform.pad_left:x_end,
    ]

    if cropped.shape != (transform.resized_height, transform.resized_width):
        raise ValueError("U-Net output padding could not be removed consistently.")

    if cropped.shape != (transform.original_height, transform.original_width):
        cropped = cv2.resize(
            cropped,
            (transform.original_width, transform.original_height),
            interpolation=cv2.INTER_NEAREST,
        )

    return np.ascontiguousarray(cropped, dtype=np.uint8)


class UNetAdapter:
    """Load the trained model once and reuse it for CPU inference requests."""

    def __init__(self):
        self._model = None
        self._device = None
        self._model_lock = Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model, self._device = poorvi_inference.load_model(
                        poorvi_inference.DEFAULT_CHECKPOINT,
                        device="cpu",
                    )
        return self._model, self._device

    def segment(self, image: Image.Image) -> np.ndarray:
        """Return a 0/255 uint8 mask for one already-decoded RGB image."""
        if image.mode != "RGB":
            raise ValueError("U-Net input must be a decoded RGB image.")

        model_input, transform = _prepare_model_input(image)
        model, device = self._get_model()
        binary_mask = poorvi_inference.predict_single_image(
            model_input,
            model,
            device,
        )
        binary_mask = np.asarray(binary_mask)

        expected_shape = (
            poorvi_inference.EXPECTED_HEIGHT,
            poorvi_inference.EXPECTED_WIDTH,
        )
        if binary_mask.shape != expected_shape:
            raise ValueError("U-Net output dimensions do not match the model input.")
        if binary_mask.dtype != np.uint8 or not np.isin(binary_mask, (0, 1)).all():
            raise ValueError("U-Net must return a uint8 binary mask with values 0 and 1.")

        restored_mask = _restore_mask(binary_mask, transform)
        return restored_mask * np.uint8(255)


_default_adapter = UNetAdapter()


def segment(image: Image.Image) -> np.ndarray:
    """Run U-Net inference through the process-wide cached adapter."""
    return _default_adapter.segment(image)
