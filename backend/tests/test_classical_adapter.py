"""Tests for the backend adapter around the Classical pipeline."""

import cv2
import numpy as np
import pytest
from PIL import Image

from backend.services import classical


def rgb_image(width=8, height=6):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = 21
    pixels[..., 1] = 82
    pixels[..., 2] = 173
    return Image.fromarray(pixels, mode="RGB")


def test_pipeline_calls_classical_stages_in_order_and_converts_rgb_to_bgr(monkeypatch):
    calls = []
    image = rgb_image()
    fov = np.full((6, 8), 255, dtype=np.uint8)
    expected_mask = np.zeros((6, 8), dtype=np.uint8)

    def extract(bgr):
        calls.append("extract")
        np.testing.assert_array_equal(bgr[0, 0], [173, 82, 21])
        return np.full((6, 8), 10, dtype=np.uint8)

    def contrast(green):
        calls.append("contrast")
        assert np.all(green == 10)
        return np.full((6, 8), 20, dtype=np.uint8)

    def blackhat(enhanced, kernel_size):
        calls.append(("blackhat", kernel_size))
        assert np.all(enhanced == 20)
        return np.full((6, 8), 30, dtype=np.uint8)

    def otsu(response, supplied_fov):
        calls.append("otsu")
        assert np.all(response == 30)
        assert supplied_fov is fov
        return expected_mask, 15.0

    monkeypatch.setattr(classical, "extract_green_channel", extract)
    monkeypatch.setattr(classical, "enhance_contrast", contrast)
    monkeypatch.setattr(classical, "enhance_vessels_blackhat", blackhat)
    monkeypatch.setattr(classical, "otsu_segmentation", otsu)

    result = classical.segment(image, fov=fov)

    assert calls == ["extract", "contrast", ("blackhat", 15), "otsu"]
    np.testing.assert_array_equal(result, expected_mask)


@pytest.mark.parametrize("size", [(3, 2), (17, 11), (1, 1)])
def test_pipeline_preserves_different_image_dimensions(monkeypatch, size):
    width, height = size
    monkeypatch.setattr(
        classical, "extract_green_channel",
        lambda image: image[:, :, 1],
    )
    monkeypatch.setattr(classical, "enhance_contrast", lambda image: image)
    monkeypatch.setattr(classical, "enhance_vessels_blackhat", lambda image, kernel_size: image)
    monkeypatch.setattr(
        classical, "otsu_segmentation",
        lambda response, fov: (np.zeros(response.shape, dtype=np.uint8), 0),
    )

    result = classical.segment(rgb_image(width, height), np.full((height, width), 255, np.uint8))

    assert result.shape == (height, width)


def test_absent_fov_uses_whole_response_otsu_fallback(monkeypatch):
    response = np.array([[0, 10], [200, 255]], dtype=np.uint8)
    monkeypatch.setattr(classical, "extract_green_channel", lambda image: image[:, :, 1])
    monkeypatch.setattr(classical, "enhance_contrast", lambda image: image)
    monkeypatch.setattr(classical, "enhance_vessels_blackhat", lambda image, kernel_size: response)
    monkeypatch.setattr(
        classical, "otsu_segmentation",
        lambda *_args: pytest.fail("FOV-aware function must not receive None"),
    )

    result = classical.segment(rgb_image(2, 2), fov=None)
    threshold, expected = cv2.threshold(
        response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    assert threshold > 0
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "bad_image",
    [None, np.zeros((4, 5, 3), dtype=np.uint8), Image.new("L", (5, 4)), Image.new("RGB", (0, 4))],
    ids=["not-pil", "numpy-instead-of-pil", "wrong-mode", "empty"],
)
def test_invalid_images_are_rejected(bad_image):
    with pytest.raises(ValueError):
        classical.segment(bad_image)


def test_invalid_fov_dimensions_are_rejected():
    with pytest.raises(ValueError, match="dimensions must match"):
        classical.segment(rgb_image(5, 4), np.ones((3, 5), dtype=np.uint8))


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((4, 5, 1), dtype=np.uint8),
        np.zeros((3, 5), dtype=np.uint8),
        np.zeros((4, 5), dtype=np.float32),
        np.full((4, 5), 1, dtype=np.uint8),
    ],
    ids=["not-2d", "wrong-size", "wrong-dtype", "not-0-255"],
)
def test_invalid_otsu_output_mask_is_rejected(monkeypatch, mask):
    monkeypatch.setattr(classical, "extract_green_channel", lambda image: image[:, :, 1])
    monkeypatch.setattr(classical, "enhance_contrast", lambda image: image)
    monkeypatch.setattr(classical, "enhance_vessels_blackhat", lambda image, kernel_size: image)
    monkeypatch.setattr(classical, "otsu_segmentation", lambda response, fov: (mask, 0))

    with pytest.raises(ValueError, match="mask"):
        classical.segment(rgb_image(5, 4), np.full((4, 5), 255, dtype=np.uint8))


def test_pipeline_exception_propagates(monkeypatch):
    def fail(_image):
        raise RuntimeError("enhancement failed")

    monkeypatch.setattr(classical, "extract_green_channel", fail)

    with pytest.raises(RuntimeError, match="enhancement failed"):
        classical.segment(Image.new("RGB", (8, 6)))
