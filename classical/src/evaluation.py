import numpy as np


def evaluate_segmentation(predicted, manual, fov):

    predicted = predicted > 0
    manual = manual > 0
    valid = fov > 0

    predicted = predicted[valid]
    manual = manual[valid]

    TP = np.sum(predicted & manual)
    TN = np.sum(~predicted & ~manual)
    FP = np.sum(predicted & ~manual)
    FN = np.sum(~predicted & manual)

    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    precision = TP / (TP + FP + 1e-8)

    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    iou = TP / (TP + FP + FN + 1e-8)

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Dice": dice,
        "IoU": iou,
        "Accuracy": accuracy
    }