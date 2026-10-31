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


def test_detailed_prediction_returns_confidence_and_probabilities(tmp_path):
    classifier = VehicleClassifier("student_model.pth")
    img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    result = classifier.predict_image_detailed(img)

    assert isinstance(result, dict)
    assert "predicted_index" in result and result["predicted_index"] in CLASS_IDX
    assert "class_name" in result and result["class_name"] == CLASS_IDX[result["predicted_index"]]
    assert "confidence" in result and 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert set(result["probabilities"].keys()) == set(CLASS_IDX.values())
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
