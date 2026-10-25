import pytest

from vehicle_classifier_core.config import load_config
from vehicle_classifier_core.reproducibility import resolve_device


def test_repository_configuration_loads():
    config = load_config("configs/distillation.yaml")
    assert config["seed"] == 1729
    assert config["training"]["checkpoint"] == "student_model.pth"


def test_missing_configuration_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot read configuration"):
        load_config(tmp_path / "missing.yaml")


def test_auto_device_falls_back_to_cpu_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_device("auto")) == "cpu"


def test_explicit_cuda_requires_cuda_capable_torch(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
