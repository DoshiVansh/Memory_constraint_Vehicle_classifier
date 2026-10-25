"""Legacy-compatible CPU inference interface."""

import logging
from pathlib import Path

import torch
from PIL import Image

from .model import CLASS_IDX, build_student_model
from .preprocessing import inference_transform

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when the model or input image cannot be used for inference."""


class VehicleClassifier:
    def __init__(self, model_path=None, preprocessing_config=None):
        self.device = torch.device("cpu")
        self.model = build_student_model(num_classes=len(CLASS_IDX)).to(self.device)
        if model_path is not None:
            checkpoint = Path(model_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            try:
                state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict, strict=True)
            except Exception as exc:
                logger.exception("Unable to load checkpoint", extra={"checkpoint": str(checkpoint)})
                raise InferenceError(f"Invalid checkpoint {checkpoint}: {exc}") from exc
        self.model.eval()
        self.transform = inference_transform(preprocessing_config)

    def predict(self, image_path: str) -> int:
        """Return the predicted class index, preserving the original public contract."""
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        except Exception as exc:
            logger.exception("Unable to read inference image", extra={"image_path": str(image_path)})
            raise InferenceError(f"Unable to read image {image_path}: {exc}") from exc
        try:
            with torch.inference_mode():
                prediction = self.model(tensor).argmax(dim=1).item()
        except Exception as exc:
            logger.exception("Model inference failed", extra={"image_path": str(image_path)})
            raise InferenceError(f"Inference failed for image {image_path}: {exc}") from exc
        return int(prediction)
