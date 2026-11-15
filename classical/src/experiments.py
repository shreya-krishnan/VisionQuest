import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.preprocessing import (
    extract_green_channel,
    enhance_contrast
)

from src.enhancement import enhance_vessels_blackhat


def measure_vessel_response(dataset, kernel_size):

    vessel_means = []
    background_means = []

    for item in dataset:

        image = item["image"]
        manual = item["manual"]
        fov = item["fov"]

        green = extract_green_channel(image)
        enhanced = enhance_contrast(green)

        response = enhance_vessels_blackhat(
            enhanced,
            kernel_size=kernel_size
        )

        vessel_pixels = (manual > 0) & (fov > 0)
        background_pixels = (manual == 0) & (fov > 0)

        vessel_mean = np.mean(response[vessel_pixels])
        background_mean = np.mean(response[background_pixels])

        vessel_means.append(vessel_mean)
        background_means.append(background_mean)

    return vessel_means, background_means


def compare_kernel_images(item):

    image = item["image"]
    fov = item["fov"]

    green = extract_green_channel(image)
    enhanced = enhance_contrast(green)

    kernel_sizes = [9, 15, 21]

    plt.figure(figsize=(16, 5))

    for i, size in enumerate(kernel_sizes):

        response = enhance_vessels_blackhat(
            enhanced,
            kernel_size=size
        )

        response = cv2.bitwise_and(
            response, response, mask=fov
        )

        plt.subplot(1, 3, i + 1)
        plt.imshow(response, cmap="gray")
        plt.title(f"Black-hat: {size}x{size}")
        plt.axis("off")

    plt.suptitle(f"Kernel Comparison - Image {item['id']}")
    plt.tight_layout()
    plt.show()
    



