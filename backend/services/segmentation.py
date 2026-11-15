"""Run selected segmentation adapters and collect independent outcomes."""

import time
from typing import Literal

import numpy as np
from PIL import Image

from backend.services import classical, unet


SegmentationMethod = Literal["classical", "unet", "both"]


def _validate_mask(mask: np.ndarray, image: Image.Image) -> None:
    expected_shape = (image.height, image.width)
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Algorithm must return a two-dimensional NumPy array.")
    if mask.shape != expected_shape:
        raise ValueError("Algorithm mask dimensions do not match the input image.")
    if mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
        raise ValueError("Algorithm mask must contain uint8 values 0 and 255.")


def segment_image(image: Image.Image, method: SegmentationMethod) -> dict:
    """Run only the requested adapter(s), in order, and report their results."""
    selected = {
        "classical": ["classical"],
        "unet": ["unet"],
        "both": ["classical", "unet"],
    }.get(method)
    if selected is None:
        raise ValueError("method must be 'classical', 'unet', or 'both'.")

    adapters = {"classical": classical.segment, "unet": unet.segment}
    outcomes = {}

    for name in selected:
        started = time.perf_counter()
        try:
            mask = adapters[name](image)
            _validate_mask(mask, image)
        except unet.UnsupportedImageDimensionsError as exc:
            outcomes[name] = {
                "status": "failed",
                "processing_time_ms": round(
                    (time.perf_counter() - started) * 1000, 3
                ),
                "error": str(exc),
            }
        except Exception:
            outcomes[name] = {
                "status": "failed",
                "processing_time_ms": round(
                    (time.perf_counter() - started) * 1000, 3
                ),
                "error": f"The {name} algorithm failed.",
            }
        else:
            outcomes[name] = {
                "status": "success",
                "processing_time_ms": round(
                    (time.perf_counter() - started) * 1000, 3
                ),
                "mask": mask,
            }

    successes = sum(outcome["status"] == "success" for outcome in outcomes.values())
    if successes == len(outcomes):
        status = "success"
    elif successes:
        status = "partial_success"
    else:
        status = "failed"

    return {"status": status, "algorithms": outcomes}
