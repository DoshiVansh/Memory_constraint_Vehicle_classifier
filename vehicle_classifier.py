"""Backward-compatible public interface for existing project scripts."""

from vehicle_classifier_core.model import CLASS_IDX, StudentShuffleNet
from vehicle_classifier_core.predictor import InferenceError, VehicleClassifier

__all__ = ["CLASS_IDX", "InferenceError", "StudentShuffleNet", "VehicleClassifier"]


if __name__ == "__main__":
    classifier = VehicleClassifier(model_path="student_model.pth")
    image_path = "test_image.jpg"
    index = classifier.predict(image_path)
    print(f"Predicted Class Index: {index}, Label: {CLASS_IDX[index]}")
