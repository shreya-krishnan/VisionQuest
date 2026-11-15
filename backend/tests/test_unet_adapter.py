"""Tests for the backend adapter around Poorvi's inference package."""

from pathlib import Path
import numpy as np
import pytest
import cv2
from PIL import Image

from backend.services import unet as unet_adapter
from backend.services.segmentation import segment_image


EXPECTED_SIZE = (565, 584)
EXPECTED_SHAPE = (584, 565)
CHECKPOINT = unet_adapter.poorvi_inference.DEFAULT_CHECKPOINT
DRIVE_IMAGE = (
    Path(__file__).resolve().parents[2]
    / "classical"
    / "DRIVE"
    / "training"
    / "images"
    / "21_training.tif"
)
def make_model_input():
    return Image.new("RGB", EXPECTED_SIZE, color=(60, 100, 140))


def test_adapter_converts_binary_prediction_to_backend_mask(monkeypatch):
    load_calls = []
    predict_calls = []
    model_input = np.asarray(make_model_input())
    binary_mask = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
    binary_mask[10, 20] = 1

    def load_model(checkpoint, device):
        load_calls.append((checkpoint, device))
        return object(), "cpu"

    def predict_single_image(rgb, model, device):
        predict_calls.append((rgb.shape, rgb.dtype, device))
        np.testing.assert_array_equal(rgb, model_input)
        return binary_mask.copy()

    monkeypatch.setattr(unet_adapter.poorvi_inference, "load_model", load_model)
    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "predict_single_image",
        predict_single_image,
    )
    adapter = unet_adapter.UNetAdapter()

    first = adapter.segment(make_model_input())
    second = adapter.segment(make_model_input())

    assert len(load_calls) == 1
    assert load_calls[0][1] == "cpu"
    assert predict_calls == [((584, 565, 3), np.uint8, "cpu")] * 2
    assert first.shape == EXPECTED_SHAPE
    assert first.dtype == np.uint8
    assert set(np.unique(first)).issubset({0, 255})
    assert first[10, 20] == 255
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("size", "resized_size", "padding"),
    [
        ((1000, 1000), (565, 565), (0, 9, 0, 10)),
        ((1200, 600), (565, 282), (0, 151, 0, 151)),
        ((400, 1000), (234, 584), (165, 0, 166, 0)),
        ((1, 1), (565, 565), (0, 9, 0, 10)),
    ],
    ids=["square", "landscape", "portrait", "very-small"],
)
def test_variable_image_sizes_preserve_aspect_pad_and_restore(
    monkeypatch, size, resized_size, padding
):
    seen = {}
    model_mask = np.ones(EXPECTED_SHAPE, dtype=np.uint8)

    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "load_model",
        lambda *args, **kwargs: (object(), "cpu"),
    )

    def predict_single_image(rgb, model, device):
        seen["input"] = rgb.copy()
        return model_mask.copy()

    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "predict_single_image",
        predict_single_image,
    )
    image = Image.new("RGB", size, color=(80, 120, 160))
    model_input, transform = unet_adapter._prepare_model_input(image)

    assert model_input.shape == (584, 565, 3)
    assert (transform.resized_width, transform.resized_height) == resized_size
    assert (
        transform.pad_left,
        transform.pad_top,
        transform.pad_right,
        transform.pad_bottom,
    ) == padding
    assert model_input[0, 0].tolist() == [0, 0, 0]
    content_y = transform.pad_top + transform.resized_height // 2
    content_x = transform.pad_left + transform.resized_width // 2
    assert model_input[content_y, content_x].tolist() != [0, 0, 0]

    output = unet_adapter.UNetAdapter().segment(image)

    assert seen["input"].shape == (584, 565, 3)
    assert output.shape == (size[1], size[0])
    assert output.dtype == np.uint8
    assert set(np.unique(output)).issubset({0, 255})
    assert np.all(output == 255)


def test_padding_is_removed_before_mask_resize(monkeypatch):
    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "load_model",
        lambda *args, **kwargs: (object(), "cpu"),
    )

    def predict_padding_only(rgb, model, device):
        mask = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
        mask[:9, :] = 1
        mask[-10:, :] = 1
        return mask

    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "predict_single_image",
        predict_padding_only,
    )

    output = unet_adapter.UNetAdapter().segment(Image.new("RGB", (1000, 1000)))

    assert output.shape == (1000, 1000)
    assert not np.any(output)


def test_restoration_uses_nearest_neighbour_for_binary_masks():
    _, transform = unet_adapter._prepare_model_input(Image.new("RGB", (100, 100)))
    prediction = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)
    y0 = transform.pad_top
    x0 = transform.pad_left
    prediction[y0:y0 + transform.resized_height,
               x0:x0 + transform.resized_width] = (
        np.indices((transform.resized_height, transform.resized_width)).sum(axis=0) % 2
    )
    content_mask = prediction[
        transform.pad_top:EXPECTED_SHAPE[0] - transform.pad_bottom,
        transform.pad_left:EXPECTED_SHAPE[1] - transform.pad_right,
    ]
    expected = cv2.resize(
        content_mask,
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    actual = unet_adapter._restore_mask(prediction, transform)

    np.testing.assert_array_equal(actual, expected)


def test_invalid_model_mask_dimensions_fail(monkeypatch):
    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "load_model",
        lambda *args, **kwargs: (object(), "cpu"),
    )
    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "predict_single_image",
        lambda *args: np.zeros((10, 10), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="dimensions do not match the model input"):
        unet_adapter.UNetAdapter().segment(Image.new("RGB", (80, 60)))


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.zeros(EXPECTED_SHAPE, dtype=np.float32),
        np.full(EXPECTED_SHAPE, 2, dtype=np.uint8),
    ],
    ids=["wrong-dtype", "non-binary-values"],
)
def test_invalid_model_mask_values_fail(monkeypatch, bad_mask):
    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "load_model",
        lambda *args, **kwargs: (object(), "cpu"),
    )
    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "predict_single_image",
        lambda *args: bad_mask,
    )

    with pytest.raises(ValueError, match="uint8 binary mask"):
        unet_adapter.UNetAdapter().segment(Image.new("RGB", (100, 100)))


def test_inference_error_is_reported_by_orchestration(monkeypatch):
    adapter = unet_adapter.UNetAdapter()

    def load_model(*args, **kwargs):
        return object(), "cpu"

    def fail_inference(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(unet_adapter.poorvi_inference, "load_model", load_model)
    monkeypatch.setattr(
        unet_adapter.poorvi_inference,
        "predict_single_image",
        fail_inference,
    )
    monkeypatch.setattr(unet_adapter, "_default_adapter", adapter)

    result = segment_image(make_model_input(), "unet")

    assert result["status"] == "failed"
    assert result["algorithms"]["unet"]["status"] == "failed"
    assert result["algorithms"]["unet"]["error"] == "The unet algorithm failed."


@pytest.mark.skipif(
    not CHECKPOINT.is_file() or not DRIVE_IMAGE.is_file(),
    reason="Trained checkpoint or DRIVE smoke-test image is unavailable.",
)
def test_real_checkpoint_inference_on_drive_image():
    adapter = unet_adapter.UNetAdapter()
    with Image.open(DRIVE_IMAGE) as source:
        image = source.convert("RGB")

    mask = adapter.segment(image)

    assert mask.shape == EXPECTED_SHAPE
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
