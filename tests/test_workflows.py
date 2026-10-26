from pathlib import Path

import yaml


WORKFLOWS_DIR = Path(__file__).parents[1] / ".github" / "workflows"


def load_workflow(name):
    return yaml.load((WORKFLOWS_DIR / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_pull_request_quality_gate_runs_fast_checks_without_training():
    workflow = load_workflow("ci.yml")
    assert "pull_request" in workflow["on"]
    assert "workflow_call" in workflow["on"]
    assert workflow["jobs"]["test-and-static-checks"]["runs-on"] == "ubuntu-latest"
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["test-and-static-checks"]["steps"])
    assert "ruff check" in commands
    assert "pytest" in commands
    assert "python train_teacher.py" not in commands
    assert "python train_distillation.py" not in commands


def test_release_workflow_requires_quality_gate_and_publishes_digest_metadata():
    workflow = load_workflow("release-image.yml")
    assert workflow["on"]["push"]["tags"] == ["v*"]
    assert workflow["jobs"]["publish"]["needs"] == "quality"
    publish = workflow["jobs"]["publish"]
    assert publish["permissions"]["packages"] == "write"
    uses = [step.get("uses", "") for step in publish["steps"]]
    assert "docker/build-push-action@v7" in uses
    assert "softprops/action-gh-release@v3" in uses
    commands = "\n".join(step.get("run", "") for step in publish["steps"])
    assert "--image-digest" in commands
    assert "--git-sha" in commands


def test_expensive_training_workflow_is_manual_and_requires_gpu_runner():
    workflow = load_workflow("train-gpu.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["jobs"]["train"]["runs-on"] == ["self-hosted", "linux", "x64", "gpu"]
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["train"]["steps"])
    assert "torch.cuda.is_available()" in commands
    assert "dvc repro train_teacher train_student evaluate_target evaluate_ood" in commands
    assert "dvc push" in commands
