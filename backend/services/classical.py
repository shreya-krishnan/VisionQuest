"""Backend adapter for the Classical retinal vessel segmentation pipeline."""

import cv2
import numpy as np
from PIL import Image

from classical.src.enhancement import enhance_vessels_blackhat
from classical.src.preprocessing import enhance_contrast, extract_green_channel
from classical.src.segmentation import otsu_segmentation


def _validate_output_mask(mask, expected_shape):
    if not isinstance(mask, np.ndarray):
        raise ValueError("Classical segmentation must return a NumPy array.")
    if mask.ndim != 2:
        raise ValueError("Classical segmentation mask must be two-dimensional.")
    if mask.shape != expected_shape:
        raise ValueError("Classical segmentation mask dimensions do not match the image.")
    if mask.dtype != np.uint8:
        raise ValueError("Classical segmentation mask must have dtype uint8.")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("Classical segmentation mask must contain only 0 and 255.")
    return mask


def segment(image: Image.Image, fov: np.ndarray | None = None) -> np.ndarray:
    """Segment one RGB PIL image and return a 0/255 uint8 mask.

    When no FOV is available, Otsu's threshold is calculated over every pixel
    in the black-hat response. This whole-image fallback is not equivalent to
    the Classical algorithm's FOV-aware thresholding.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Classical input must be a PIL image.")
    if image.mode != "RGB":
        raise ValueError("Classical input image must be RGB.")

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Classical input must have shape (H, W, 3).")
    if rgb.dtype != np.uint8:
        raise ValueError("Classical input image must have dtype uint8.")
    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Classical input image dimensions must be non-empty.")

    if fov is not None:
        if not isinstance(fov, np.ndarray) or fov.ndim != 2:
            raise ValueError("FOV must be a two-dimensional NumPy array.")
        if fov.shape != (height, width):
            raise ValueError("FOV dimensions must match the input image.")
        if fov.dtype != np.uint8:
            raise ValueError("FOV must have dtype uint8.")
        if not np.isin(fov, (0, 255)).all():
            raise ValueError("FOV must contain only 0 and 255.")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    green = extract_green_channel(bgr)
    enhanced = enhance_contrast(green)
    response = enhance_vessels_blackhat(enhanced, kernel_size=15)

    if fov is not None:
        result = otsu_segmentation(response, fov)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("otsu_segmentation must return (mask, threshold).")
        mask, _threshold = result
    else:
        # FOV-free fallback: threshold all response pixels. This can include
        # background outside the retinal field because no FOV is available.
        _threshold, mask = cv2.threshold(
            response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return _validate_output_mask(mask, (height, width))
