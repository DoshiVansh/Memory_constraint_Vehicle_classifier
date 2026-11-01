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
    assert "scripts/model_gate.py" in commands
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
    assert "dispatches" in commands


def test_expensive_training_workflow_is_manual_and_requires_gpu_runner():
    workflow = load_workflow("train-gpu.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["jobs"]["train"]["runs-on"] == ["self-hosted", "linux", "x64", "gpu"]
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["train"]["steps"])
    assert "torch.cuda.is_available()" in commands
    assert "dvc repro train_teacher train_student evaluate_target evaluate_ood" in commands
    assert "dvc push" in commands
    assert 'access_key_id "${DAGSHUB_TOKEN}"' in commands


def test_kubernetes_hpa_and_deployment_ha():
    k8s_dir = Path(__file__).parents[1] / "k8s"
    hpa_file = k8s_dir / "hpa.yaml"
    assert hpa_file.is_file(), "k8s/hpa.yaml must exist"
    hpa_spec = yaml.safe_load(hpa_file.read_text(encoding="utf-8"))
    assert hpa_spec["kind"] == "HorizontalPodAutoscaler"
    assert hpa_spec["spec"]["minReplicas"] == 2
    assert hpa_spec["spec"]["maxReplicas"] >= 5
    assert hpa_spec["spec"]["scaleTargetRef"]["name"] == "vehicle-classifier-api"

    dep_file = k8s_dir / "deployment.yaml.tpl"
    dep_content = dep_file.read_text(encoding="utf-8")
    assert "replicas: 2" in dep_content

    jenkinsfile = Path(__file__).parents[1] / "Jenkinsfile"
    jenkins_content = jenkinsfile.read_text(encoding="utf-8")
    assert "RELEASE_TAG" in jenkins_content
    assert "k8s/hpa.yaml" in jenkins_content


def test_kubernetes_monitoring_alerts_and_pvc():
    monitoring_dir = Path(__file__).parents[1] / "k8s" / "monitoring"
    prom_file = monitoring_dir / "prometheus.yaml"
    assert prom_file.is_file(), "k8s/monitoring/prometheus.yaml must exist"
    prom_docs = list(yaml.safe_load_all(prom_file.read_text(encoding="utf-8")))

    config_cm = next(d for d in prom_docs if d.get("kind") == "ConfigMap")
    rules_text = config_cm["data"]["vehicle-classifier.yml"]
    assert "VehicleClassifierHighInferenceLatency" in rules_text
    assert "VehicleClassifierLowConfidenceSpike" in rules_text
    assert "vehicle-classifier-alertmanager" in config_cm["data"]["prometheus.yml"]

    pvc_doc = next(d for d in prom_docs if d.get("kind") == "PersistentVolumeClaim")
    assert pvc_doc["metadata"]["name"] == "vehicle-classifier-prometheus-pvc"
    assert pvc_doc["spec"]["resources"]["requests"]["storage"] == "10Gi"

    alertmanager_file = monitoring_dir / "alertmanager.yaml"
    assert alertmanager_file.is_file(), "k8s/monitoring/alertmanager.yaml must exist"
    am_docs = list(yaml.safe_load_all(alertmanager_file.read_text(encoding="utf-8")))
    assert any(d.get("kind") == "Deployment" and d["metadata"]["name"] == "vehicle-classifier-alertmanager" for d in am_docs)
    assert any(d.get("kind") == "Service" and d["metadata"]["name"] == "vehicle-classifier-alertmanager" for d in am_docs)
    assert any(d.get("kind") == "PersistentVolumeClaim" and d["metadata"]["name"] == "vehicle-classifier-alertmanager-pvc" for d in am_docs)


def test_cloud_cd_workflow_verifies_staging_and_smoke_tests():
    workflow = load_workflow("deploy.yml")
    assert "workflow_run" in workflow["on"]
    assert "repository_dispatch" in workflow["on"]
    assert "workflow_dispatch" in workflow["on"]
    job = workflow["jobs"]["staging-deploy-and-smoke-test"]
    assert job["runs-on"] == "ubuntu-latest"
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "validate_release_metadata.py" in commands
    assert "kubectl apply --dry-run=client" in commands
    assert "pytest" in commands

