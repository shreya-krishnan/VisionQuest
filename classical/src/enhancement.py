
import cv2
import numpy as np


def enhance_vessels_blackhat(image, kernel_size=15):
    """
    Highlight dark structures using morphological
    black-hat transformation.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        image,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    return blackhat
