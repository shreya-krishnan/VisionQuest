"""Tests for segmentation placeholders, orchestration, and the API route."""

import base64
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.services import classical, unet
from backend.services.segmentation import segment_image


client = TestClient(app)


def make_rgb_image(width=8, height=6):
    return Image.new("RGB", (width, height), color=(50, 100, 150))


def encode_png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def upload_segment(method="both", image_bytes=None):
    return client.post(
        "/segment",
        data={"method": method},
        files={
            "file": (
                "retina.png",
                image_bytes if image_bytes is not None else encode_png(make_rgb_image()),
                "image/png",
            )
        },
    )


def test_classical_placeholder_returns_deterministic_binary_mask():
    image = make_rgb_image()
    first = classical.segment(image)
    second = classical.segment(image)

    assert first.shape == (image.height, image.width)
    assert first.dtype == np.uint8
    assert set(np.unique(first)).issubset({0, 255})
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("method", "expected_call"),
    [("classical", "classical"), ("unet", "unet")],
)
def test_orchestration_calls_only_selected_algorithm(
    monkeypatch, method, expected_call
):
    called = []

    def classical_stub(image):
        called.append("classical")
        return np.zeros((image.height, image.width), dtype=np.uint8)

    def unet_stub(image):
        called.append("unet")
        return np.zeros((image.height, image.width), dtype=np.uint8)

    monkeypatch.setattr(classical, "segment", classical_stub)
    monkeypatch.setattr(unet, "segment", unet_stub)

    result = segment_image(make_rgb_image(), method)

    assert called == [expected_call]
    assert list(result["algorithms"]) == [expected_call]
    assert result["status"] == "success"
    assert result["algorithms"][expected_call]["processing_time_ms"] >= 0


def test_both_algorithms_run_sequentially(monkeypatch):
    called = []

    def stub(name):
        def run(image):
            called.append(name)
            return np.zeros((image.height, image.width), dtype=np.uint8)
        return run

    monkeypatch.setattr(classical, "segment", stub("classical"))
    monkeypatch.setattr(unet, "segment", stub("unet"))

    result = segment_image(make_rgb_image(), "both")

    assert called == ["classical", "unet"]
    assert result["status"] == "success"
    assert set(result["algorithms"]) == {"classical", "unet"}
    assert all(
        outcome["processing_time_ms"] >= 0
        for outcome in result["algorithms"].values()
    )


def test_partial_success_survives_other_algorithm_failure(monkeypatch):
    def fail(_image):
        raise RuntimeError("private implementation detail")

    monkeypatch.setattr(classical, "segment", fail)
    monkeypatch.setattr(
        unet,
        "segment",
        lambda image: np.zeros((image.height, image.width), dtype=np.uint8),
    )

    result = segment_image(make_rgb_image(), "both")

    assert result["status"] == "partial_success"
    assert result["algorithms"]["classical"]["status"] == "failed"
    assert result["algorithms"]["classical"]["error"] == (
        "The classical algorithm failed."
    )
    assert result["algorithms"]["unet"]["status"] == "success"


def test_partial_success_when_unet_fails_and_classical_succeeds(monkeypatch):
    def fail(_image):
        raise RuntimeError("private implementation detail")

    monkeypatch.setattr(unet, "segment", fail)

    result = segment_image(make_rgb_image(), "both")

    assert result["status"] == "partial_success"
    assert result["algorithms"]["unet"]["status"] == "failed"
    assert result["algorithms"]["unet"]["error"] == "The unet algorithm failed."
    assert result["algorithms"]["classical"]["status"] == "success"


def test_all_algorithm_failures_return_failed_result(monkeypatch):
    def fail(_image):
        raise RuntimeError("failure")

    monkeypatch.setattr(classical, "segment", fail)
    monkeypatch.setattr(unet, "segment", fail)

    result = segment_image(make_rgb_image(), "both")

    assert result["status"] == "failed"
    assert all(
        outcome["status"] == "failed"
        for outcome in result["algorithms"].values()
    )


def test_segment_endpoint_returns_png_data_urls(monkeypatch):
    monkeypatch.setattr(
        unet,
        "segment",
        lambda image: np.zeros((image.height, image.width), dtype=np.uint8),
    )
    response = upload_segment("both")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "retina.png"
    assert (body["width"], body["height"]) == (8, 6)
    assert body["method"] == "both"
    assert body["status"] == "success"
    assert set(body["algorithms"]) == {"classical", "unet"}

    for outcome in body["algorithms"].values():
        assert outcome["status"] == "success"
        assert outcome["processing_time_ms"] >= 0
        prefix, encoded = outcome["mask_data_url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        with Image.open(BytesIO(base64.b64decode(encoded))) as mask:
            assert mask.format == "PNG"
            assert mask.mode == "L"
            assert mask.size == (8, 6)
            assert set(np.unique(np.asarray(mask))).issubset({0, 255})


def test_segment_endpoint_returns_partial_success(monkeypatch):
    def fail(_image):
        raise RuntimeError("failure")

    monkeypatch.setattr(classical, "segment", fail)
    monkeypatch.setattr(
        unet,
        "segment",
        lambda image: np.zeros((image.height, image.width), dtype=np.uint8),
    )

    response = upload_segment("both")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial_success"
    assert body["algorithms"]["classical"]["status"] == "failed"
    assert body["algorithms"]["unet"]["status"] == "success"
    assert "mask_data_url" not in body["algorithms"]["classical"]


def test_segment_endpoint_returns_500_when_all_algorithms_fail(monkeypatch):
    def fail(_image):
        raise RuntimeError("failure")

    monkeypatch.setattr(classical, "segment", fail)
    monkeypatch.setattr(unet, "segment", fail)

    response = upload_segment("both")

    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "failed"
    assert all(
        outcome["status"] == "failed"
        for outcome in body["algorithms"].values()
    )


def test_segment_rejects_invalid_method():
    response = upload_segment("random")

    assert response.status_code == 422


def test_segment_reuses_upload_validation():
    response = upload_segment("classical", b"not an image")

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not read the uploaded image."
