
import cv2
import numpy as np


def otsu_segmentation(response, fov):

    # Select only pixels inside the retinal region
    valid_pixels = response[fov > 0].reshape(-1, 1)

    # Calculate Otsu's optimal threshold
    threshold_value, _ = cv2.threshold(
        valid_pixels,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Apply the calculated threshold to the complete image
    _, segmented = cv2.threshold(
        response,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Remove pixels outside the field of view
    segmented = cv2.bitwise_and(
        segmented,
        segmented,
        mask=fov
    )

    return segmented, threshold_value

def adaptive_segmentation(response, fov, block_size=51, C=2):

    segmented = cv2.adaptiveThreshold(
        response,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # Remove pixels outside the retinal region
    segmented = cv2.bitwise_and(
        segmented,
        segmented,
        mask=fov
    )

    return segmented