# Retinal Vessel Segmentation — U-Net Inference

Standalone inference package extracted from the supplied Colab notebook and matched to `best_unet.pth`.

## Files

- `model.py`: exact `DoubleConv` and `UNet` architecture used for training.
- `inference.py`: checkpoint loading and single-image inference; no Colab, Drive, dataset, or training setup.
- `test_inference.py`: inference smoke test.
- `checkpoints/best_unet.pth`: supplied trained checkpoint (best epoch 24).

## Environment

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Install a PyTorch build suitable for your machine from https://pytorch.org/get-started/locally/ if the default pip installation is not appropriate for your CPU/GPU.

## Exact input preprocessing

The training notebook's dataset loader:
1. Reads the fundus image with OpenCV.
2. Converts OpenCV BGR to RGB.
3. Converts to `float32` and divides by 255.0.
4. Transposes HWC to CHW.
5. Adds a batch dimension for inference.

Expected image: RGB, `584 x 565 x 3`, `uint8` in `[0, 255]`.
No resizing, cropping, per-image standardization, or color-channel normalization is performed by this inference module. Wrong dimensions are rejected to avoid silently changing the training input geometry.

## Inference

```python
from inference import load_model, predict_single_image

model, device = load_model("checkpoints/best_unet.pth")
binary_mask = predict_single_image(
    "path/to/DRIVE_image.png",
    model,
    device,
    threshold=0.5,
)

print(binary_mask.shape)  # (584, 565)
print(binary_mask.dtype)  # uint8
print(set(binary_mask.flatten().tolist()))  # subset of {0, 1}
```

Or use the convenience function:

```python
from inference import load_and_predict

binary_mask = load_and_predict("path/to/DRIVE_image.png")
```

`predict_single_image` and `load_and_predict` return a NumPy `uint8` binary mask with values `0` and `1`. Multiply by 255 only if a display/save format requires 0/255.

## Threshold and FOV handling

- Sigmoid is applied to the model logits.
- Default threshold is `0.5`, matching the baseline validation/export run.
- The inference module does not apply an FOV mask automatically. The training loss and reported evaluation were FOV-masked, but the network itself predicts over the full image. If a downstream evaluation requires FOV restriction, apply the corresponding binary FOV mask separately: `prediction = prediction * (fov_mask > 0)`.

## Test

With a real DRIVE image:

```bash
python test_inference.py path/to/21_training.tif
```

Without an argument:

```bash
python test_inference.py
```

The no-argument test uses a synthetic RGB image. It verifies checkpoint loading, inference execution, output dimensions, dtype, and binary values; it does not measure segmentation accuracy.

## Checkpoint provenance

The included checkpoint was supplied as `best_unet.pth`. Its stored metadata reports:
- Best epoch: 24
- Validation Dice used for checkpoint selection: `0.7084033933532993`

The model state dictionary loads strictly into the architecture in `model.py`.
