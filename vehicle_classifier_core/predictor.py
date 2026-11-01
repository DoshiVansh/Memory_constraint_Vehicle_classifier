"""Legacy-compatible CPU inference interface."""

import logging
from pathlib import Path

import torch
from PIL import Image

from .model import CLASS_IDX, build_student_model
from .preprocessing import inference_transform
from .telemetry import get_tracer

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
                decoded = image.convert("RGB")
        except Exception as exc:
            logger.exception("Unable to read inference image", extra={"image_path": str(image_path)})
            raise InferenceError(f"Unable to read image {image_path}: {exc}") from exc
        return self.predict_image(decoded)

    def predict_image_detailed(self, image: Image.Image) -> dict:
        """Predict from an already decoded PIL image, returning class index, name, confidence, and probabilities."""
        tracer = get_tracer()
        try:
            with tracer.start_as_current_span("tensor_preprocessing") as span:
                tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
                span.set_attribute("tensor.shape", str(list(tensor.shape)))
                span.set_attribute("tensor.device", str(self.device))
        except Exception as exc:
            raise InferenceError(f"Unable to preprocess image: {exc}") from exc
        try:
            with tracer.start_as_current_span("pytorch_forward_pass") as span:
                with torch.inference_mode():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    predicted_index = int(probs.argmax().item())
                    confidence = float(probs[predicted_index].item())
                    probabilities = {
                        CLASS_IDX[i]: float(probs[i].item())
                        for i in range(len(CLASS_IDX))
                    }
                span.set_attribute("model.num_classes", len(CLASS_IDX))
                span.set_attribute("prediction.class_name", CLASS_IDX[predicted_index])
                span.set_attribute("prediction.confidence", round(confidence, 6))
        except Exception as exc:
            logger.exception("Model inference failed")
            raise InferenceError(f"Inference failed: {exc}") from exc
        return {
            "predicted_index": predicted_index,
            "class_name": CLASS_IDX[predicted_index],
            "confidence": round(confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in probabilities.items()},
        }

    def predict_image(self, image: Image.Image) -> int:
        """Predict from an already decoded PIL image."""
        return self.predict_image_detailed(image)["predicted_index"]
