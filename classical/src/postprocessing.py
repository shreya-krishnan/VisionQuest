
import cv2
import numpy as np


def remove_small_components(segmented, fov, min_size=5):
    """
    Remove small isolated components from a binary vessel image.
    """

    # Label all connected white regions
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            segmented,
            connectivity=8
        )
    )

    # Create an empty output image
    cleaned = np.zeros_like(segmented)

    # Label 0 represents the background, so skip it
    for label in range(1, num_labels):

        # Get the area of the current component
        area = stats[label, cv2.CC_STAT_AREA]

        # Retain components meeting the minimum area
        if area >= min_size:
            cleaned[labels == label] = 255

    # Ensure pixels outside the FOV remain black
    cleaned = cv2.bitwise_and(cleaned, cleaned, mask=fov)

    return cleaned
