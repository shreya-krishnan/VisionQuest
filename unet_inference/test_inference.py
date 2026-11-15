"""Smoke test: python test_inference.py [path/to/DRIVE_image.png]

Without an image argument, tests with a synthetic RGB uint8 image of the
required dimensions. That fallback verifies loading/inference/output shape,
not segmentation quality on a real retina.
"""
import sys
from pathlib import Path

import numpy as np

try:
    from .inference import load_and_predict, EXPECTED_HEIGHT, EXPECTED_WIDTH
except ImportError:
    from inference import load_and_predict, EXPECTED_HEIGHT, EXPECTED_WIDTH


def main():
    if len(sys.argv) > 1:
        image = sys.argv[1]
        print(f"Testing on image: {image}")
    else:
        # Synthetic smoke-test input; replace with a DRIVE fundus image
        # for a meaningful visual/segmentation check.
        image = np.zeros((EXPECTED_HEIGHT, EXPECTED_WIDTH, 3), dtype=np.uint8)
        image[:, :, 0] = 80
        image[:, :, 1] = 100
        image[:, :, 2] = 120
        print("No image supplied; using synthetic RGB smoke-test input.")

    mask = load_and_predict(image)

    assert isinstance(mask, np.ndarray), "Output is not a NumPy array"
    assert mask.shape == (EXPECTED_HEIGHT, EXPECTED_WIDTH), mask.shape
    assert mask.dtype == np.uint8, mask.dtype
    assert set(np.unique(mask).tolist()).issubset({0, 1})

    print("Inference smoke test passed.")
    print("Output shape:", mask.shape)
    print("Output dtype:", mask.dtype)
    print("Unique mask values:", np.unique(mask))
    print("Foreground pixels:", int(mask.sum()))


if __name__ == "__main__":
    main()
