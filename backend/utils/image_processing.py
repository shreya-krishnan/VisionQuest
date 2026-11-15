"""Upload validation and image decoding helpers."""

import io
import warnings

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "TIFF"}


class ImageDecodeError(ValueError):
    """Raised when an upload is empty, unsupported, or invalid image data."""


class ImageTooLargeError(ValueError):
    """Raised when an upload exceeds the configured byte limit."""


def _validate_dimensions(image: Image.Image) -> None:
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ImageDecodeError("Image exceeds the 20 million pixel limit.")


async def decode_uploaded_image(upload: UploadFile) -> tuple[Image.Image, str]:
    """Validate image bytes and return an RGB image and its original format."""
    contents = await upload.read(MAX_UPLOAD_BYTES + 1)

    if len(contents) > MAX_UPLOAD_BYTES:
        raise ImageTooLargeError
    if not contents:
        raise ImageDecodeError("The uploaded file is empty.")

    try:
        # Pillow warnings are errors only while this upload is being decoded.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(io.BytesIO(contents)) as probe:
                image_format = probe.format
                if image_format not in SUPPORTED_FORMATS:
                    raise ImageDecodeError(
                        "Only JPEG, PNG and TIFF images are supported."
                    )
                _validate_dimensions(probe)
                probe.verify()

            # verify() checks the file structure; reopen it to decode pixels.
            with Image.open(io.BytesIO(contents)) as source:
                _validate_dimensions(source)
                source.load()
                rgb_image = source.convert("RGB")
    except ImageDecodeError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ImageDecodeError(
            "Image exceeds the 20 million pixel limit."
        ) from None
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        EOFError,
    ):
        raise ImageDecodeError("Could not read the uploaded image.") from None

    return rgb_image, image_format
