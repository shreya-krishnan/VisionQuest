
from pathlib import Path
import re
import cv2


def get_image_id(filename):
    """Extract the numerical image ID from a filename."""

    match = re.match(r"(\d+)", filename)

    if match:
        return match.group(1)

    return None


def get_file_map(folder):
    """Create a mapping of image IDs to file paths."""

    folder = Path(folder)

    files = {}

    for path in folder.iterdir():

        if not path.is_file():
            continue

        image_id = get_image_id(path.name)

        if image_id is not None:

            if image_id in files:
                raise ValueError(
                    f"Duplicate image ID {image_id} found in {folder}"
                )

            files[image_id] = path

    return files


def load_training_dataset(dataset_dir):

    dataset_dir = Path(dataset_dir)

    image_dir = dataset_dir / "images"
    manual_dir = dataset_dir / "1st_manual"
    fov_dir = dataset_dir / "mask"

    images = get_file_map(image_dir)
    manuals = get_file_map(manual_dir)
    fovs = get_file_map(fov_dir)

    dataset = []

    for image_id in sorted(images, key=int):

        if image_id not in manuals or image_id not in fovs:
            print(f"Missing corresponding file for image {image_id}")
            continue

        image = cv2.imread(
            str(images[image_id]),
            cv2.IMREAD_COLOR
        )

        manual = cv2.imread(
            str(manuals[image_id]),
            cv2.IMREAD_GRAYSCALE
        )

        fov = cv2.imread(
            str(fovs[image_id]),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or manual is None or fov is None:
            print(f"Error loading image ID: {image_id}")
            continue

        if image.shape[:2] != manual.shape or image.shape[:2] != fov.shape:
            print(f"Dimension mismatch for image ID: {image_id}")
            continue

        dataset.append({
            "id": image_id,
            "image": image,
            "manual": manual,
            "fov": fov
        })

    return dataset


def load_test_dataset(dataset_dir):

    dataset_dir = Path(dataset_dir)

    image_dir = dataset_dir / "images"
    fov_dir = dataset_dir / "mask"

    images = get_file_map(image_dir)
    fovs = get_file_map(fov_dir)

    dataset = []

    for image_id in sorted(images, key=int):

        if image_id not in fovs:
            print(f"Missing FOV mask for image {image_id}")
            continue

        image = cv2.imread(
            str(images[image_id]),
            cv2.IMREAD_COLOR
        )

        fov = cv2.imread(
            str(fovs[image_id]),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or fov is None:
            print(f"Error loading image ID: {image_id}")
            continue

        if image.shape[:2] != fov.shape:
            print(f"Dimension mismatch for image ID: {image_id}")
            continue

        dataset.append({
            "id": image_id,
            "image": image,
            "fov": fov
        })

    return dataset
