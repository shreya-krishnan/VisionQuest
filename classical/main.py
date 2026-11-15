
from pathlib import Path
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_training_dataset, load_test_dataset
from src.preprocessing import (
    extract_green_channel,
    enhance_contrast,
    remove_noise
)
from src.enhancement import enhance_vessels_blackhat
from src.segmentation import otsu_segmentation
from src.evaluation import evaluate_segmentation

BASE_DIR = Path(__file__).resolve().parent

DRIVE_DIR = BASE_DIR / "DRIVE"
TRAIN_DIR = DRIVE_DIR / "training"
TEST_DIR = DRIVE_DIR / "test"
OUTPUT_DIR = BASE_DIR / "outputs" / "training"

def create_output_folders():

    folders = [
        "green_channel",
        "enhanced",
        "filtered",
        "vessel_enhanced"
    ]

    for folder in folders:
        (OUTPUT_DIR / folder).mkdir(
            parents=True,
            exist_ok=True
        )

def process_training_dataset(dataset):

    create_output_folders()

    for item in dataset:

        image_id = item["id"]
        image = item["image"]
        fov = item["fov"]

        # Stage 1: Green channel extraction
        green = extract_green_channel(image)

        # Stage 2: Contrast enhancement
        enhanced = enhance_contrast(green)

        # Stage 3: Noise reduction
        filtered = remove_noise(enhanced)

        # Stage 4: Vessel enhancement
        vessel_enhanced = enhance_vessels_blackhat(enhanced)

        # Restrict outputs to the valid retinal region
        green = cv2.bitwise_and(green, green, mask=fov)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=fov)
        filtered = cv2.bitwise_and(filtered, filtered, mask=fov)
        vessel_enhanced = cv2.bitwise_and(
        vessel_enhanced, vessel_enhanced, mask=fov
    )

        # Save each intermediate output
        cv2.imwrite(
            str(OUTPUT_DIR / "green_channel" / f"{image_id}.png"),
            green
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "enhanced" / f"{image_id}.png"),
            enhanced
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "filtered" / f"{image_id}.png"),
            filtered
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "vessel_enhanced" / f"{image_id}.png"),
            vessel_enhanced
        )

        # print(f"Processed image: {image_id}")

def process_test_dataset(dataset):

    segmentation_dir = BASE_DIR / "outputs" / "test" / "segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # print("\n========== TEST DATASET SEGMENTATION ==========")

    for item in dataset:

        image_id = item["id"]
        image = item["image"]
        fov = item["fov"]

        # Stage 1: Green channel extraction
        green = extract_green_channel(image)

        # Stage 2: CLAHE enhancement
        enhanced = enhance_contrast(green)

        # Stage 3: Black-hat vessel enhancement
        response = enhance_vessels_blackhat(
            enhanced,
            kernel_size=15
        )

        # Stage 4: Otsu segmentation
        segmented, threshold_value = otsu_segmentation(
            response,
            fov
        )

        # Save the predicted vessel mask
        output_path = segmentation_dir / f"{image_id}.png"

        cv2.imwrite(str(output_path), segmented)

        # print(
        #     f"Image {image_id} processed | "
        #     f"Otsu threshold: {threshold_value}"
        # )

    print("\nTest segmentation completed.")
    print("Saved masks to:", segmentation_dir)

def display_results(item):

    image = item["image"]
    image_id = item["id"]

    green = extract_green_channel(image)
    enhanced = enhance_contrast(green)
    filtered = remove_noise(enhanced)
    vessel_enhanced = enhance_vessels_blackhat(enhanced)

    fov = item["fov"]

    green = cv2.bitwise_and(green, green, mask=fov)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=fov)
    filtered = cv2.bitwise_and(filtered, filtered, mask=fov)
    vessel_enhanced = cv2.bitwise_and(
        vessel_enhanced, vessel_enhanced, mask=fov
    )

    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        green,
        enhanced,
        filtered,
        vessel_enhanced
    ]

    titles = [
        "Original Image",
        "Green Channel",
        "CLAHE Enhanced",
        "Median Filtered",
        "Black-hat Enhanced"
    ]

    plt.figure(figsize=(18, 5))

    for i, (img, title) in enumerate(zip(images, titles)):

        plt.subplot(1, 5, i + 1)

        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap="gray")

        plt.title(title)
        plt.axis("off")

    plt.suptitle(f"Preprocessing - Image {image_id}")
    plt.tight_layout()
    plt.show()

    # Compare intensity histograms inside the FOV
    plt.figure(figsize=(10, 5))

    plt.hist(green[fov > 0].ravel(), bins=256,
             alpha=0.5, label="Green Channel")

    plt.hist(enhanced[fov > 0].ravel(), bins=256,
             alpha=0.5, label="CLAHE Enhanced")

    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.title("Intensity Distribution Before and After CLAHE")
    plt.legend()
    plt.show()

def evaluate_full_dataset(dataset):

    metric_names = [
        "Sensitivity",
        "Specificity",
        "Precision",
        "Dice",
        "IoU",
        "Accuracy"
    ]

    all_metrics = []
    csv_rows = []

    # Folder for saving binary segmentation masks
    segmentation_dir = OUTPUT_DIR / "segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # print("\n========== FULL DATASET EVALUATION ==========")

    for item in dataset:

        image_id = item["id"]
        image = item["image"]
        manual = item["manual"]
        fov = item["fov"]

        # Stage 1: Green channel extraction
        green = extract_green_channel(image)

        # Stage 2: CLAHE enhancement
        enhanced = enhance_contrast(green)

        # Stage 3: Black-hat vessel enhancement
        response = enhance_vessels_blackhat(
            enhanced,
            kernel_size=15
        )

        # Stage 4: Otsu segmentation
        segmented, threshold_value = otsu_segmentation(
            response,
            fov
        )

        # Stage 5: Save segmentation mask
        mask_path = segmentation_dir / f"{image_id}.png"

        saved = cv2.imwrite(str(mask_path), segmented)

        if not saved:
            raise OSError(f"Could not save segmentation mask: {mask_path}")

        # Stage 6: Evaluate inside FOV
        metrics = evaluate_segmentation(
            segmented,
            manual,
            fov
        )

        metric_values = [
            metrics[name] for name in metric_names
        ]

        all_metrics.append(metric_values)

        # Store this image's results for the CSV
        csv_rows.append([
            image_id,
            threshold_value,
            *metric_values
        ])

        # print(f"\nImage ID: {image_id}")
        # print(f"Otsu threshold: {threshold_value}")

        # for name in metric_names:
        #     print(f"{name}: {metrics[name]:.4f}")

    # Convert results into a NumPy array
    all_metrics = np.array(all_metrics)

    # Calculate mean and standard deviation
    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)

    print("\n========== FINAL DATASET SUMMARY ==========")
    print(f"Total images evaluated: {len(dataset)}")

    for i, name in enumerate(metric_names):

        print(
            f"{name}: "
            f"Mean = {mean_metrics[i]:.4f}, "
            f"Std = {std_metrics[i]:.4f}"
        )

    # Add summary rows to CSV
    csv_rows.append(["MEAN", "", *mean_metrics])
    csv_rows.append(["STD", "", *std_metrics])

    # Write all results to CSV
    csv_path = OUTPUT_DIR / "evaluation_metrics.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Image_ID",
            "Otsu_Threshold",
            *metric_names
        ])

        writer.writerows(csv_rows)

    print(f"\nSegmentation masks saved in: {segmentation_dir}")
    print(f"Evaluation results saved in: {csv_path}")

    return all_metrics

def inspect_challenging_images(dataset):

    target_ids = {"23", "30", "34"}

    for item in dataset:

        if item["id"] not in target_ids:
            continue

        image = item["image"]
        manual = item["manual"]
        fov = item["fov"]

        green = extract_green_channel(image)
        enhanced = enhance_contrast(green)

        response = enhance_vessels_blackhat(
            enhanced,
            kernel_size=15
        )

        predicted, threshold = otsu_segmentation(response, fov)

        manual_binary = manual > 0
        predicted_binary = predicted > 0
        valid = fov > 0

        # Create a colour-coded error visualization
        overlay = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        tp = predicted_binary & manual_binary & valid
        fp = predicted_binary & ~manual_binary & valid
        fn = ~predicted_binary & manual_binary & valid

        overlay[tp] = [0, 255, 0]    # Correct detection
        overlay[fp] = [255, 0, 0]    # False positive
        overlay[fn] = [0, 0, 255]    # Missed vessel

        plt.figure(figsize=(16, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        ))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(manual, cmap="gray")
        plt.title("Manual Ground Truth")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(predicted, cmap="gray")
        plt.title("Otsu Prediction")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(overlay)
        plt.title("Error Analysis")
        plt.axis("off")

        plt.suptitle(
            f"Image {item['id']} | Threshold: {threshold}"
        )

        plt.tight_layout()
        plt.show()

def main():

    # Step 1: Load training dataset
    dataset = load_training_dataset(TRAIN_DIR)

    print("Total training images:", len(dataset))

    if not dataset:
        print("No training images found.")
        return

    # Step 2: Save preprocessing outputs
    process_training_dataset(dataset)

    # Step 3: Generate and evaluate Otsu segmentation
    evaluate_full_dataset(dataset)

    print("\nFinal training dataset evaluation completed!")

    test_dataset = load_test_dataset(TEST_DIR)

    print("\n========== TEST DATASET SUMMARY ==========")
    print("Total test images:", len(test_dataset))

    # Generate test segmentation masks
    process_test_dataset(test_dataset)

if __name__ == "__main__":
    main()
