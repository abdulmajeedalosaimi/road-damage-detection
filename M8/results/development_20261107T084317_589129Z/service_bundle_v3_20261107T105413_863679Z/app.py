
import base64
import hashlib
import io
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, UploadFile
from ultralytics import YOLO

from pipeline import run_integrated_inference

BUNDLE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
INFERENCE_LOCK = threading.Lock()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def get_project_root():
    configured = os.environ.get("M8_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()

    launch = json.loads(
        (BUNDLE_DIR / "launch.local.json").read_text(encoding="utf-8")
    )
    return (
        BUNDLE_DIR / launch["project_root_relative_to_bundle"]
    ).resolve()


@asynccontextmanager
async def lifespan(app):
    app.state.ready = False

    config = json.loads(
        (BUNDLE_DIR / "config.json").read_text(encoding="utf-8")
    )
    project_root = get_project_root()
    models = {}

    for name in ("m4", "m5"):
        checkpoint = (project_root / config[name]["checkpoint"]).resolve()

        if not checkpoint.is_relative_to(project_root):
            raise ValueError("Checkpoint path must remain inside project root.")

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {checkpoint}")

        if sha256(checkpoint) != config[name]["sha256"]:
            raise ValueError(f"{name} checkpoint fingerprint mismatch.")

        models[name] = YOLO(str(checkpoint))
        models[name].to(config["device"])

    if models["m4"].task != "detect":
        raise ValueError("M4 must be a detector.")
    if models["m5"].task != "segment":
        raise ValueError("M5 must be a segmenter.")
    if dict(models["m4"].names) != {
        0: "D00", 1: "D10", 2: "D20", 3: "D40"
    }:
        raise ValueError("Unexpected M4 classes.")
    if dict(models["m5"].names) != {0: "pothole"}:
        raise ValueError("Unexpected M5 classes.")

    app.state.config = config
    app.state.models = models
    app.state.ready = True

    try:
        yield
    finally:
        app.state.ready = False
        app.state.models.clear()


app = FastAPI(
    title="M8 Road Damage Inference",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Models are not ready.")

    config = app.state.config
    return {
        "status": "ready",
        "device": config["device"],
        "m4_sha256": config["m4"]["sha256"],
        "m5_sha256": config["m5"]["sha256"],
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_image_pixels": MAX_IMAGE_PIXELS,
    }


def decode_image(payload):
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=415,
                    detail="Only JPEG and PNG images are supported.",
                )

            width, height = image.size
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=400, detail="Invalid dimensions.")

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413, detail="Image exceeds pixel limit."
                )

            if getattr(image, "n_frames", 1) != 1:
                raise HTTPException(
                    status_code=415, detail="Animated images are not supported."
                )

            # Use encoded pixel orientation, consistent with the notebook.
            image.load()
            rgb = np.array(image.convert("RGB"))

    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="Image exceeds pixel limit.")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400, detail="File is not a readable image."
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def png_base64(array):
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise RuntimeError("PNG encoding failed.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.post("/predict")
def predict(file: UploadFile):
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Models are not ready.")

    try:
        payload = file.file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        file.file.close()

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds 10 MiB.")

    image_bgr = decode_image(payload)
    config = app.state.config

    # Models are shared within this server process.
    with INFERENCE_LOCK:
        report, annotated, mask = run_integrated_inference(
            image_bgr=image_bgr,
            detector=app.state.models["m4"],
            segmenter=app.state.models["m5"],
            detector_settings=config["m4"]["settings"],
            segmenter_settings=config["m5"]["settings"],
            device=config["device"],
        )

    return {
        "report": report,
        "model_fingerprints": {
            "m4": config["m4"]["sha256"],
            "m5": config["m5"]["sha256"],
        },
        "annotated_png_base64": png_base64(annotated),
        "mask_png_base64": png_base64(mask.astype(np.uint8) * 255),
    }
