"""DRIVE validation-subset evaluation using the production inference adapters."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.services import classical, unet


VALIDATION_IDS = (26, 30, 37, 40)
DATASET_DIR = Path(__file__).resolve().parents[2] / "classical" / "DRIVE" / "training"
VALIDATION_LABEL = (
    "DRIVE validation subset — 4 images (IDs 26, 30, 37, 40). "
    "The U-Net checkpoint was selected using validation Dice; these results "
    "are validation performance, not independent test performance."
)


class EvaluationDataError(ValueError):
    """Raised when a validation image or its matching references are unusable."""


def _read_gray(path: Path, kind: str, image_id: int) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise EvaluationDataError(f"Could not read {kind} for DRIVE ID {image_id}: {path.name}.")
    return value


def load_validation_item(image_id: int, dataset_dir: Path = DATASET_DIR):
    """Load a validation sample by its explicit DRIVE ID; never rely on order."""
    if image_id not in VALIDATION_IDS:
        raise EvaluationDataError(f"DRIVE ID {image_id} is not in the validation subset.")
    image_path = dataset_dir / "images" / f"{image_id}_training.tif"
    annotation_path = dataset_dir / "1st_manual" / f"{image_id}_manual1.gif"
    fov_path = dataset_dir / "mask" / f"{image_id}_training_mask.gif"
    for path, kind in ((image_path, "source image"), (annotation_path, "first-observer annotation"), (fov_path, "FOV mask")):
        if not path.is_file():
            raise EvaluationDataError(f"Missing {kind} for DRIVE ID {image_id}: {path.name}.")

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise EvaluationDataError(f"Could not read source image for DRIVE ID {image_id}.") from exc
    annotation = _read_gray(annotation_path, "first-observer annotation", image_id)
    fov = _read_gray(fov_path, "FOV mask", image_id)
    expected = (image.height, image.width)
    if annotation.shape != expected:
        raise EvaluationDataError(f"Annotation dimensions for DRIVE ID {image_id} do not match its source image.")
    if fov.shape != expected:
        raise EvaluationDataError(f"FOV dimensions for DRIVE ID {image_id} do not match its source image.")
    return image, annotation, fov


def calculate_counts(prediction: np.ndarray, annotation: np.ndarray, fov: np.ndarray) -> dict:
    """Count binary outcomes inside FOV; thresholds are >0 and >127 respectively."""
    if not all(isinstance(array, np.ndarray) and array.ndim == 2 for array in (prediction, annotation, fov)):
        raise EvaluationDataError("Prediction, annotation and FOV must be two-dimensional arrays.")
    if prediction.shape != annotation.shape or prediction.shape != fov.shape:
        raise EvaluationDataError("Prediction, annotation and FOV dimensions must match exactly.")
    predicted_vessels = prediction > 0
    reference_vessels = annotation > 127
    valid = fov > 127
    tp = int(np.count_nonzero(predicted_vessels & reference_vessels & valid))
    tn = int(np.count_nonzero(~predicted_vessels & ~reference_vessels & valid))
    fp = int(np.count_nonzero(predicted_vessels & ~reference_vessels & valid))
    fn = int(np.count_nonzero(~predicted_vessels & reference_vessels & valid))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "valid_pixels": int(valid.sum())}


def metrics_from_counts(counts: dict) -> dict:
    """Return None for a metric whose denominator is zero."""
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
    }


def _validated_prediction(mask: np.ndarray, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != expected_shape:
        raise EvaluationDataError(f"{name} prediction dimensions do not match the source image.")
    return mask


def run_dataset_evaluation(dataset_dir: Path = DATASET_DIR) -> dict:
    """Evaluate both production algorithms on the same four validation images."""
    per_image = []
    pooled = {name: {key: 0 for key in ("tp", "tn", "fp", "fn", "valid_pixels")} for name in ("classical", "unet")}
    for image_id in VALIDATION_IDS:
        image, annotation, fov = load_validation_item(image_id, Path(dataset_dir))
        expected = (image.height, image.width)
        binary_fov = np.where(fov > 127, 255, 0).astype(np.uint8)
        predictions = {
            "classical": _validated_prediction(classical.segment(image, fov=binary_fov), expected, "Classical"),
            "unet": _validated_prediction(unet.segment(image), expected, "U-Net"),
        }
        row = {"id": image_id, "width": image.width, "height": image.height, "algorithms": {}}
        for name, prediction in predictions.items():
            counts = calculate_counts(prediction, annotation, fov)
            row["algorithms"][name] = {**counts, "metrics": metrics_from_counts(counts)}
            for key, value in counts.items():
                pooled[name][key] += value
        per_image.append(row)
    return {
        "status": "success",
        "label": VALIDATION_LABEL,
        "dataset": "DRIVE",
        "subset": "validation",
        "ids": list(VALIDATION_IDS),
        "aggregation": "pooled_counts",
        "methodology": {
            "prediction_threshold": "pixel value > 0 is vessel (production masks are binary 0/255)",
            "annotation_threshold": "grayscale value > 127 is vessel",
            "fov_threshold": "grayscale value > 127 is valid",
            "zero_denominator": "metric is null when its denominator is zero",
            "scope": "Counts include only pixels inside the matching FOV mask.",
        },
        "algorithms": {
            name: {**counts, "metrics": metrics_from_counts(counts)}
            for name, counts in pooled.items()
        },
        "per_image": per_image,
    }
