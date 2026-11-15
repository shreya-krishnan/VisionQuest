from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.services import evaluation


client = TestClient(app)


def test_known_metric_counts_and_ratios():
    prediction = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    annotation = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    fov = np.full((2, 2), 255, dtype=np.uint8)
    counts = evaluation.calculate_counts(prediction, annotation, fov)
    assert counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1, "valid_pixels": 4}
    assert evaluation.metrics_from_counts(counts) == {
        "dice": 0.5, "iou": 1 / 3, "sensitivity": 0.5, "specificity": 0.5,
    }


def test_fov_excludes_outside_pixels():
    prediction = np.array([[255, 255]], dtype=np.uint8)
    annotation = np.array([[0, 255]], dtype=np.uint8)
    fov = np.array([[0, 255]], dtype=np.uint8)
    assert evaluation.calculate_counts(prediction, annotation, fov) == {
        "tp": 1, "tn": 0, "fp": 0, "fn": 0, "valid_pixels": 1,
    }


def test_zero_denominators_return_null_metrics():
    counts = evaluation.calculate_counts(
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    )
    assert evaluation.metrics_from_counts(counts) == {
        "dice": None, "iou": None, "sensitivity": None, "specificity": None,
    }


def test_invalid_metric_dimensions_rejected():
    with pytest.raises(evaluation.EvaluationDataError, match="dimensions must match"):
        evaluation.calculate_counts(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)))


def test_missing_annotation_rejected(tmp_path):
    for folder in ("images", "mask", "1st_manual"):
        (tmp_path / folder).mkdir()
    Image.new("RGB", (5, 4)).save(tmp_path / "images" / "26_training.tif")
    Image.new("L", (5, 4)).save(tmp_path / "mask" / "26_training_mask.gif")
    with pytest.raises(evaluation.EvaluationDataError, match="Missing first-observer annotation"):
        evaluation.load_validation_item(26, tmp_path)


def test_validation_files_are_paired_by_id(tmp_path):
    for folder in ("images", "mask", "1st_manual"):
        (tmp_path / folder).mkdir()
    Image.new("RGB", (5, 4)).save(tmp_path / "images" / "26_training.tif")
    Image.new("L", (5, 4), 255).save(tmp_path / "1st_manual" / "26_manual1.gif")
    Image.new("L", (5, 4), 255).save(tmp_path / "mask" / "26_training_mask.gif")
    image, annotation, fov = evaluation.load_validation_item(26, tmp_path)
    assert image.size == (5, 4)
    assert annotation.shape == fov.shape == (4, 5)
    with pytest.raises(evaluation.EvaluationDataError, match="not in the validation subset"):
        evaluation.load_validation_item(27, tmp_path)


def test_both_algorithms_run_on_identical_validation_ids(monkeypatch):
    calls = []
    image = Image.new("RGB", (2, 2))
    annotation = np.zeros((2, 2), dtype=np.uint8)
    fov = np.full((2, 2), 255, dtype=np.uint8)
    monkeypatch.setattr(evaluation, "load_validation_item", lambda image_id, _root: (calls.append(("load", image_id)) or image, annotation, fov))

    def classical_stub(supplied_image, fov=None):
        calls.append(("classical", supplied_image, fov))
        return np.zeros((2, 2), dtype=np.uint8)

    def unet_stub(supplied_image):
        calls.append(("unet", supplied_image))
        return np.zeros((2, 2), dtype=np.uint8)

    monkeypatch.setattr(evaluation.classical, "segment", classical_stub)
    monkeypatch.setattr(evaluation.unet, "segment", unet_stub)
    result = evaluation.run_dataset_evaluation()
    assert [row["id"] for row in result["per_image"]] == [26, 30, 37, 40]
    assert [entry[0] for entry in calls].count("classical") == 4
    assert [entry[0] for entry in calls].count("unet") == 4
    for index, image_id in enumerate((26, 30, 37, 40)):
        load_call, classical_call, unet_call = calls[index * 3:index * 3 + 3]
        assert load_call == ("load", image_id)
        assert classical_call[1] is unet_call[1]
        assert classical_call[2] is not None
    assert result["label"].startswith("DRIVE validation subset — 4 images")


def test_pooled_metrics_use_summed_counts(monkeypatch):
    image = Image.new("RGB", (2, 2))
    annotation = np.zeros((2, 2), dtype=np.uint8)
    fov = np.full((2, 2), 255, dtype=np.uint8)
    monkeypatch.setattr(evaluation, "load_validation_item", lambda *_args: (image, annotation, fov))
    monkeypatch.setattr(evaluation.classical, "segment", lambda *_args, **_kwargs: np.array([[255, 0], [0, 0]], dtype=np.uint8))
    monkeypatch.setattr(evaluation.unet, "segment", lambda *_args: np.zeros((2, 2), dtype=np.uint8))
    result = evaluation.run_dataset_evaluation()
    classical = result["algorithms"]["classical"]
    assert classical["fp"] == 4
    assert classical["tn"] == 12
    assert classical["metrics"]["specificity"] == 12 / 16


def test_api_returns_success_and_clear_data_errors(monkeypatch):
    expected = {"status": "success", "algorithms": {}, "per_image": []}
    monkeypatch.setattr("backend.main.run_dataset_evaluation", lambda: expected)
    assert client.post("/evaluation/dataset").json() == expected
    monkeypatch.setattr("backend.main.run_dataset_evaluation", lambda: (_ for _ in ()).throw(evaluation.EvaluationDataError("missing annotation")))
    response = client.post("/evaluation/dataset")
    assert response.status_code == 422
    assert response.json()["detail"] == "missing annotation"
    monkeypatch.setattr("backend.main.run_dataset_evaluation", lambda: (_ for _ in ()).throw(RuntimeError("private error")))
    response = client.post("/evaluation/dataset")
    assert response.status_code == 500
    assert response.json()["detail"] == "Dataset evaluation failed while running both algorithms."
