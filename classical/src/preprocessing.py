
import cv2


def extract_green_channel(image):
    """Extract the green channel from a BGR retinal image."""
    return image[:, :, 1]


def enhance_contrast(green, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhance local contrast using CLAHE."""

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced = clahe.apply(green)

    return enhanced


def remove_noise(enhanced, kernel_size=3):
    """Apply median filtering to reduce small intensity variations."""

    filtered = cv2.medianBlur(enhanced, kernel_size)

    return filtered
