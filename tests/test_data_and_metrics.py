from PIL import Image

from vehicle_classifier_core.data import make_train_validation_loaders
from vehicle_classifier_core.evaluation import classification_report


def _make_dataset(root):
    for index, name in enumerate(("0_Bus", "1_Truck", "2_Car", "3_Bike", "4_None")):
        class_dir = root / name
        class_dir.mkdir(parents=True)
        for sample in range(4):
            Image.new("RGB", (12, 12), color=(index * 30, sample * 20, 10)).save(
                class_dir / f"{sample}.png"
            )


def test_dataset_split_is_repeatable(tmp_path):
    dataset = tmp_path / "data"
    _make_dataset(dataset)
    first_train, first_validation = make_train_validation_loaders(
        dataset, batch_size=4, validation_fraction=0.25, seed=17
    )
    second_train, second_validation = make_train_validation_loaders(
        dataset, batch_size=4, validation_fraction=0.25, seed=17
    )

    assert first_train.dataset.indices == second_train.dataset.indices
    assert first_validation.dataset.indices == second_validation.dataset.indices


def test_classification_report_uses_true_rows_and_prediction_columns():
    result = classification_report([0, 0, 1, 2], [0, 1, 1, 2], num_classes=3)

    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert result["per_class"]["0"]["support"] == 2
    assert result["per_class"]["0"]["correct"] == 1
    assert result["per_class"]["0"]["accuracy"] == 0.5
    assert result["per_class"]["0"]["precision"] == 1.0
    assert result["per_class"]["0"]["recall"] == 0.5
    assert round(result["per_class"]["0"]["f1_score"], 4) == round(2 / 3, 4)
    assert round(result["macro_f1"], 4) == round(7 / 9, 4)
    assert result["weighted_f1"] == 0.75
