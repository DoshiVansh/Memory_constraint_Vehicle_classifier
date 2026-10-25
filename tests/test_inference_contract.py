from PIL import Image

from vehicle_classifier import CLASS_IDX, VehicleClassifier


def test_legacy_prediction_returns_stable_class_index(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24), color=(40, 120, 200)).save(image_path)
    classifier = VehicleClassifier()

    first = classifier.predict(str(image_path))
    second = classifier.predict(str(image_path))

    assert isinstance(first, int)
    assert first in CLASS_IDX
    assert second == first


def test_canonical_checkpoint_loads_and_predicts():
    classifier = VehicleClassifier("student_model.pth")
    assert classifier.model.training is False
    assert next(classifier.model.parameters()).device.type == "cpu"
