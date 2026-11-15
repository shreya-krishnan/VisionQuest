"""Tests for the VisionQuest upload and health endpoints."""

import asyncio
import struct
import zlib
from io import BytesIO

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.utils.image_processing import (
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    decode_uploaded_image,
)


client = TestClient(app)


def make_image(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color=(40, 120, 200)).save(
        buffer, format=image_format
    )
    return buffer.getvalue()


def make_image_in_mode(mode: str, image_format: str, pixel) -> bytes:
    image = Image.new(mode, (2, 2), color=pixel)
    if mode == "P":
        palette = [12, 34, 56] + [0, 0, 0] * 255
        image.putpalette(palette)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def make_png_with_dimensions(width: int, height: int) -> bytes:
    """Build a PNG header with large dimensions without allocating its pixels."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00"))
        + _png_chunk(b"IEND", b"")
    )


def decode_bytes(contents: bytes):
    upload = UploadFile(filename="test-image", file=BytesIO(contents))
    return asyncio.run(decode_uploaded_image(upload))


@pytest.mark.parametrize(
    ("image_format", "filename", "content_type"),
    [
        ("TIFF", "retina.tif", "image/tiff"),
        ("PNG", "retina.png", "image/png"),
        ("JPEG", "retina.jpg", "image/jpeg"),
    ],
)
def test_upload_valid_images(image_format, filename, content_type):
    response = client.post(
        "/upload",
        files={"file": (filename, make_image(image_format), content_type)},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": filename,
        "message": "Image uploaded successfully",
        "width": 8,
        "height": 6,
        "format": image_format,
        "status": "success",
    }


def test_png_content_is_accepted_despite_misleading_metadata():
    response = client.post(
        "/upload",
        files={"file": ("photo.gif", make_image("PNG"), "image/gif")},
    )

    assert response.status_code == 200
    assert response.json()["format"] == "PNG"
    assert response.json()["filename"] == "photo.gif"


def test_unsupported_content_is_rejected_despite_allowed_metadata():
    response = client.post(
        "/upload",
        files={"file": ("photo.png", make_image("GIF"), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only JPEG, PNG and TIFF images are supported."
    )


def test_home_returns_expected_response():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to VisionQuest API",
        "status": "running",
    }


def test_health_returns_healthy_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_empty_upload_is_rejected():
    response = client.post(
        "/upload", files={"file": ("empty.png", b"", "image/png")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_corrupted_image_is_rejected():
    response = client.post(
        "/upload",
        files={"file": ("broken.png", b"not a real image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not read the uploaded image."


def test_truncated_supported_image_is_rejected():
    png_data = make_image("PNG")
    response = client.post(
        "/upload",
        files={"file": ("truncated.png", png_data[:-12], "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not read the uploaded image."


def test_image_over_pixel_limit_is_rejected_before_full_decode():
    width = 5000
    height = (MAX_IMAGE_PIXELS // width) + 1
    response = client.post(
        "/upload",
        files={
            "file": (
                "large-dimensions.png",
                make_png_with_dimensions(width, height),
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Image exceeds the 20 million pixel limit."
    )


def test_pillow_decompression_bomb_warning_is_rejected(monkeypatch):
    # Lower Pillow's threshold for this test so a small fixture emits its warning.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 30)
    response = client.post(
        "/upload",
        files={"file": ("image.png", make_image("PNG"), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Image exceeds the 20 million pixel limit."
    )


@pytest.mark.parametrize(
    ("mode", "image_format", "pixel", "expected_rgb"),
    [
        ("L", "PNG", 73, (73, 73, 73)),
        ("P", "PNG", 0, (12, 34, 56)),
        ("CMYK", "TIFF", (0, 255, 255, 0), (255, 0, 0)),
    ],
)
def test_helper_converts_supported_modes_to_rgb(
    mode, image_format, pixel, expected_rgb
):
    image, detected_format = decode_bytes(
        make_image_in_mode(mode, image_format, pixel)
    )

    assert image.mode == "RGB"
    assert detected_format == image_format
    assert image.getpixel((0, 0)) == expected_rgb


def test_unsupported_image_format_is_rejected():
    response = client.post(
        "/upload",
        files={"file": ("animated.gif", make_image("GIF"), "image/gif")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only JPEG, PNG and TIFF images are supported."
    )


def test_oversized_upload_is_rejected():
    response = client.post(
        "/upload",
        files={
            "file": (
                "large.png",
                b"x" * (MAX_UPLOAD_BYTES + 1),
                "image/png",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image exceeds the 10 MB upload limit."


def test_missing_file_is_rejected():
    response = client.post("/upload")

    assert response.status_code == 422
