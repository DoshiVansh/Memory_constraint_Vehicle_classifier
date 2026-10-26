"""FastAPI inference service for the vehicle classifier."""

import logging
import os
import warnings
import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, Info, make_asgi_app
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

REQUESTS = Counter(
    "vehicle_classifier_requests_total",
    "HTTP requests handled by the inference service.",
    ("method", "endpoint", "status"),
)
ERRORS = Counter(
    "vehicle_classifier_errors_total",
    "HTTP errors grouped into bounded status classes.",
    ("endpoint", "error_type"),
)
REQUEST_DURATION = Histogram(
    "vehicle_classifier_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("endpoint",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
IN_PROGRESS = Gauge(
    "vehicle_classifier_in_progress_requests",
    "Inference API requests currently being handled.",
    ("endpoint",),
)
INFERENCE_DURATION = Histogram(
    "vehicle_classifier_inference_duration_seconds",
    "Time spent in the model prediction call, excluding image upload and decode.",
    ("model_version",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
PREDICTIONS = Counter(
    "vehicle_classifier_predictions_total",
    "Successful predictions by fixed class name and loaded model version.",
    ("predicted_class", "model_version"),
)
MODEL_INFO = Info("vehicle_classifier_model", "Identity of the loaded classifier model.")
KNOWN_ENDPOINTS = {"/predict", "/health/live", "/health/ready"}
IGNORED_METRIC_PATHS = {"/metrics", "/health/live", "/health/ready"}


class PrometheusMetricsMiddleware:
    """Record HTTP metrics using route templates and a fixed label vocabulary."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in IGNORED_METRIC_PATHS:
            await self.app(scope, receive, send)
            return

        raw_path = scope.get("path", "")
        endpoint = raw_path if raw_path in KNOWN_ENDPOINTS else "unmatched"
        method = scope.get("method", "OTHER")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            method = "OTHER"
        started = time.perf_counter()
        status_code = 500
        IN_PROGRESS.labels(endpoint=endpoint).inc()

        async def capture_status(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            elapsed = time.perf_counter() - started
            IN_PROGRESS.labels(endpoint=endpoint).dec()
            REQUESTS.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
            REQUEST_DURATION.labels(endpoint=endpoint).observe(elapsed)
            if status_code >= 500:
                ERRORS.labels(endpoint=endpoint, error_type="http_5xx").inc()
            elif status_code >= 400:
                ERRORS.labels(endpoint=endpoint, error_type="http_4xx").inc()


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
    app.state.model_version = package["model_version"]
    MODEL_INFO.info({"model_version": package["model_version"], "architecture": package["architecture"]})
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
app.add_middleware(PrometheusMetricsMiddleware)
app.mount("/metrics", make_asgi_app())


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
        model_version = getattr(request.app.state, "model_version", "unknown")
        inference_started = time.perf_counter()
        try:
            prediction = request.app.state.classifier.predict_image(rgb)
        finally:
            INFERENCE_DURATION.labels(model_version=model_version).observe(
                time.perf_counter() - inference_started
            )
        PREDICTIONS.labels(
            predicted_class=CLASS_IDX[prediction], model_version=model_version
        ).inc()
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
