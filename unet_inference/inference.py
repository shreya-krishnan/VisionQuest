"""Standalone single-image inference for the trained retinal U-Net.

Input:
  - RGB uint8 NumPy array, shape (584, 565, 3), OR
  - path to an image file (OpenCV reads it and converts BGR -> RGB).

Preprocessing exactly matches training:
  RGB float32 / 255.0, HWC -> CHW, add batch dimension.
Output:
  uint8 NumPy binary mask, shape (584, 565), values {0, 1}.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

try:
    from .model import UNet
except ImportError:  # allows running inference.py directly from this folder
    from model import UNet


EXPECTED_HEIGHT = 584
EXPECTED_WIDTH = 565
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best_unet.pth"


def load_model(
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: Union[str, torch.device, None] = None,
) -> tuple:
    """Load the matching trained checkpoint. Returns (model, device)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=True
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


def _read_rgb_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                "A NumPy image must be an RGB array with shape (H, W, 3)."
            )
        if rgb.dtype != np.uint8:
            raise TypeError("NumPy input must be uint8 with values in [0, 255].")

    if rgb.shape[:2] != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise ValueError(
            f"Expected image dimensions {(EXPECTED_HEIGHT, EXPECTED_WIDTH)}, "
            f"received {rgb.shape[:2]}. This model was trained at 584x565; "
            "the inference module does not resize images."
        )
    return rgb


def predict_single_image(
    image: Union[str, Path, np.ndarray],
    model: torch.nn.Module,
    device: Union[str, torch.device],
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Return a (584, 565) uint8 NumPy mask containing only 0 and 1."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    rgb = _read_rgb_image(image)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.sigmoid(logits)
        binary = (probabilities >= threshold).to(torch.uint8)

    mask = binary.squeeze(0).squeeze(0).cpu().numpy()
    return mask


def load_and_predict(
    image: Union[str, Path, np.ndarray],
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    device: Union[str, torch.device, None] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Convenience function: load checkpoint and predict one image."""
    model, device = load_model(checkpoint_path, device)
    return predict_single_image(image, model, device, threshold)
