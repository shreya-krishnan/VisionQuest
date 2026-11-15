
import base64
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from backend.utils.image_processing import (
    ImageDecodeError,
    ImageTooLargeError,
    decode_uploaded_image,
)
from backend.services.segmentation import segment_image
from backend.services.evaluation import EvaluationDataError, run_dataset_evaluation

app = FastAPI(
    title="VisionQuest API",
    description="Retinal Blood Vessel Segmentation",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to VisionQuest API",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/evaluation/dataset")
def evaluate_drive_validation():
    """Evaluate the fixed four-image DRIVE validation subset."""
    try:
        return run_dataset_evaluation()
    except EvaluationDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Dataset evaluation failed while running both algorithms.") from exc


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        image, image_format = await decode_uploaded_image(file)

        return {
            "filename": file.filename,
            "message": "Image uploaded successfully",
            "width": image.width,
            "height": image.height,
            "format": image_format,
            "status": "success"
        }
    except ImageTooLargeError:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit."
        )
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


@app.post("/segment")
async def segment_upload(
    file: UploadFile = File(...),
    method: Literal["classical", "unet", "both"] = Form(...),
):
    try:
        image, _ = await decode_uploaded_image(file)
    except ImageTooLargeError:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit."
        )
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = segment_image(image, method)
    algorithms = {}

    for name, outcome in result["algorithms"].items():
        algorithm_result = {
            "status": outcome["status"],
            "processing_time_ms": outcome["processing_time_ms"],
        }
        if outcome["status"] == "success":
            mask_buffer = BytesIO()
            Image.fromarray(outcome["mask"]).save(mask_buffer, format="PNG")
            encoded_mask = base64.b64encode(mask_buffer.getvalue()).decode("ascii")
            algorithm_result["mask_data_url"] = (
                f"data:image/png;base64,{encoded_mask}"
            )
        else:
            algorithm_result["error"] = outcome["error"]
        algorithms[name] = algorithm_result

    response = {
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "method": method,
        "status": result["status"],
        "algorithms": algorithms,
    }

    if result["status"] == "failed":
        return JSONResponse(status_code=500, content=response)
    return response
