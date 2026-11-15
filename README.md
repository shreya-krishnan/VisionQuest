# VisionQuest — Retinal Vessel Segmentation

VisionQuest is a retinal vessel segmentation project that explores two approaches to identifying blood vessels in retinal fundus images: classical image processing and deep learning using U-Net.

The project provides a web interface where users can upload retinal images, run either segmentation approach, compare their outputs, and evaluate segmentation performance against expert annotations from the DRIVE dataset.

## 1. Project Objectives

- Segment blood vessels from retinal fundus images.
- Implement and compare a classical image processing pipeline and a U-Net model.
- Provide a simple interface for uploading images and viewing segmentation results.
- Evaluate both approaches using standard segmentation metrics.
- Study the strengths and limitations of classical and deep learning methods.

## 2. System Architecture

VisionQuest consists of a React frontend and a FastAPI backend.

### Frontend
The frontend is developed using React and Vite. It provides:
- Image upload and segmentation method selection.
- Classical, U-Net, and combined segmentation options.
- Display of original images and predicted vessel masks.
- Results comparison.
- Dataset-based evaluation and metric visualisation.
- A methodology section explaining the approaches.

### Backend
The backend is developed using FastAPI. It:
- Receives uploaded images through API requests.
- Validates and decodes the input image.
- Routes requests to the selected segmentation service.
- Collects the generated masks and processing information.
- Returns the results to the frontend.

The Classical and U-Net implementations are integrated as Python services within the backend.

### Overall workflow

1. The user uploads a retinal image.
2. The frontend sends the image and selected method to the backend.
3. FastAPI routes the request to the appropriate segmentation service.
4. The selected algorithm processes the image and generates a binary vessel mask.
5. The backend returns the result to the frontend.
6. The frontend displays the original image and segmentation output.

## 3. Segmentation Approaches

### A. Classical Image Processing

The classical pipeline uses image processing operations to identify retinal vessels.

The integrated pipeline includes:
1. Green channel extraction.
2. Contrast enhancement.
3. Black-hat filtering for vessel enhancement.
4. Otsu thresholding for binary segmentation.
5. Field-of-view masking when the corresponding mask is available.

The output is a binary vessel mask.

### B. U-Net

U-Net is a convolutional neural network designed for image segmentation.

The implementation uses an encoder-decoder architecture with skip connections.

The inference workflow includes:
1. Image preprocessing and normalisation.
2. Conversion to the model's expected input format.
3. U-Net prediction.
4. Sigmoid activation and thresholding.
5. Generation of a binary segmentation mask.

The trained checkpoint is stored at:

`unet_inference/checkpoints/best_unet.pth`

Git LFS is used to track this model checkpoint.

## 4. Dataset

VisionQuest uses the DRIVE (Digital Retinal Images for Vessel Extraction) dataset.

Official dataset website:

https://drive.grand-challenge.org/DRIVE/

The local project expects the following directory structure:

```text
classical/
└── DRIVE/
    ├── training/
    │   ├── images/
    │   ├── 1st_manual/
    │   └── mask/
    └── test/
        ├── images/
        └── mask/
```

The DRIVE dataset is not included in this GitHub repository. Each collaborator must obtain it from the official source and place the extracted files in the expected directory structure.

The training portion contains 20 labelled images. In this project, 16 images are used for training and four images (IDs 26, 30, 37 and 40) are used for validation.

The official test images do not have publicly supplied vessel annotations in the local dataset copy used for this project.

## 5. Evaluation

The evaluation module compares predicted vessel masks with expert manual annotations.

The following metrics are calculated:

- **Dice Coefficient:** Measures the overlap between the predicted and ground-truth vessel masks.
- **Intersection over Union (IoU):** Measures the ratio of the intersection to the union of the masks.
- **Sensitivity:** Measures how many actual vessel pixels are correctly identified.
- **Specificity:** Measures how many background pixels are correctly identified.

Evaluation is performed within the field-of-view region.

### Validation results

The following are pooled results from the four labelled validation images:

| Metric | Classical | U-Net |
|---|---:|---:|
| Dice | 0.6863 | 0.7084 |
| IoU | 0.5224 | 0.5485 |
| Sensitivity | 0.6604 | 0.6071 |
| Specificity | 0.9646 | 0.9857 |

U-Net achieved higher Dice, IoU and specificity on this validation subset. Classical processing achieved higher sensitivity.

These results represent validation performance, not independent test performance. The U-Net checkpoint was selected using validation Dice, and the evaluation uses the same four validation images.

## 6. Project Structure

```text
Retinal_vessel_project/
├── backend/
│   ├── main.py
│   ├── services/
│   ├── tests/
│   └── utils/
├── classical/
│   ├── main.py
│   ├── src/
│   └── DRIVE/                 # Download separately
├── frontend/
│   ├── src/
│   ├── img/
│   └── package.json
├── unet/
├── unet_inference/
│   ├── checkpoints/
│   ├── inference.py
│   └── model.py
├── .gitattributes
└── .gitignore
```

## 7. Installation and Execution

### Prerequisites

- Python with compatible project dependencies.
- Node.js and npm.
- Git.
- DRIVE dataset files for dataset-based evaluation.

### Step 1: Clone the repository

```bash
git clone https://github.com/shreya-krishnan/VisionQuest.git
cd VisionQuest
```

### Step 2: Set up the Python environment

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies using the requirements files provided in the project:

```bash
pip install -r backend/requirements-dev.txt
pip install -r classical/requirements.txt
pip install -r unet_inference/requirements.txt
```

If the model checkpoint is not present after cloning, install Git LFS and retrieve it:

```bash
git lfs install
git lfs pull
```

### Step 3: Start the backend

Run this command from the project root:

```bash
python -m uvicorn backend.main:app --reload
```

The backend should be available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### Step 4: Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually:

`http://localhost:5173`

Keep both the frontend and backend running while using the application.

## 8. API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /segment` | Performs image segmentation using the selected method. |
| `POST /evaluation/dataset` | Evaluates the methods on the configured DRIVE validation subset. |

The segmentation endpoint supports `classical`, `unet`, and `both` modes.

## 9. Limitations

- Evaluation is currently based on four labelled validation images.
- The U-Net checkpoint was selected using validation Dice, so the reported results are not independent test results.
- The local DRIVE test set does not contain the expert vessel annotations required for independent test metrics.
- The segmentation methods are designed for retinal fundus images; arbitrary photographs are not guaranteed to produce meaningful results.
- Performance on other retinal datasets has not yet been established.

## 10. Future Improvements

- Evaluate on an independent labelled retinal dataset.
- Expand the validation and testing process.
- Improve input validation for non-retinal images.
- Explore model optimisation and segmentation quality improvements.
- Extend the comparison to additional segmentation techniques.

## Technologies Used

- Python
- FastAPI
- React.js
- Vite
- OpenCV
- NumPy
- PyTorch
- U-Net
- Git and Git LFS

---

**VisionQuest — Retinal Vessel Segmentation using Classical Image Processing and U-Net.**
