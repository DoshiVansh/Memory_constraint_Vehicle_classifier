"""FastAPI inference service for the vehicle classifier."""

import logging
import os
import warnings
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.responses import JSONResponse

from .artifact_validation import validate_inference_package
from .model import CLASS_IDX
from .predictor import InferenceError, VehicleClassifier

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(20_000_000)))
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
if MAX_UPLOAD_BYTES <= 0 or MAX_IMAGE_PIXELS <= 0:
    raise ValueError("MAX_UPLOAD_BYTES and MAX_IMAGE_PIXELS must be positive integers")
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class PredictionResponse(BaseModel):
    predicted_index: int
    class_name: str
    model_sha256: str


class ErrorResponse(BaseModel):
    detail: str


class RequestBodySizeLimit:
    """Cap the whole HTTP body, including multipart framing and streamed requests."""

    def __init__(self, app, max_body_bytes):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for key, value in scope.get("headers", []) if key.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
                await response(scope, receive, send)
                return
            if declared_length < 0:
                response = JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
                await response(scope, receive, send)
                return
            if declared_length > self.max_body_bytes:
                response = JSONResponse(status_code=413, content={"detail": "Request body exceeds the upload limit"})
                await response(scope, receive, send)
                return

        total_bytes = 0

        async def limited_receive():
            nonlocal total_bytes
            message = await receive()
            if message["type"] == "http.request":
                total_bytes += len(message.get("body", b""))
                if total_bytes > self.max_body_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            response = JSONResponse(status_code=413, content={"detail": "Request body exceeds the upload limit"})
            await response(scope, receive, send)


class RequestBodyTooLarge(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "student_model.pth")
    metadata_path = os.getenv("MODEL_METADATA_PATH", f"{model_path}.json")
    package = validate_inference_package(model_path, metadata_path)
    # Strict state-dict loading uses the existing CPU predictor. The model factory
    # uses weights=None, so startup has no pretrained-weight download path.
    app.state.classifier = VehicleClassifier(model_path=model_path)
    app.state.model_sha256 = package["sha256"]
    logger.info("Inference package validated", extra={"model_sha256": package["sha256"]})
    yield
    app.state.classifier = None


app = FastAPI(
    title="Vehicle Classifier API",
    version="1.0.0",
    description="CPU inference for the five-class vehicle classifier.",
    lifespan=lifespan,
)


app.add_middleware(RequestBodySizeLimit, max_body_bytes=MAX_UPLOAD_BYTES + 256 * 1024)


@app.get("/health/live", tags=["health"])
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready", tags=["health"])
async def readiness(request: Request):
    if getattr(request.app.state, "classifier", None) is None:
        raise HTTPException(status_code=503, detail="Model package is not ready")
    return {"status": "ready", "model_sha256": request.app.state.model_sha256}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Malformed image or request"},
        413: {"model": ErrorResponse, "description": "Upload or pixel limit exceeded"},
        415: {"model": ErrorResponse, "description": "Unsupported image format"},
        500: {"model": ErrorResponse, "description": "Inference failed"},
    },
    tags=["inference"],
)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = bytearray()
        while True:
            chunk = await file.read(min(1024 * 1024, MAX_UPLOAD_BYTES + 1 - len(contents)))
            if not chunk:
                break
            contents.extend(chunk)
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES} byte limit")
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded image is empty")

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise HTTPException(status_code=415, detail="Image must be JPEG, PNG, or WEBP")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="Image dimensions exceed the pixel limit")
                image.verify()
        # Reopen verified bytes; predictor owns RGB conversion and preprocessing.
        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            rgb = image.convert("RGB")
        prediction = request.app.state.classifier.predict_image(rgb)
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid supported image") from exc
    except InferenceError as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed") from exc
    finally:
        await file.close()

    return PredictionResponse(
        predicted_index=prediction,
        class_name=CLASS_IDX[prediction],
        model_sha256=request.app.state.model_sha256,
    )
