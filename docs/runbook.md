# Operator runbook

This guide is the practical path from checkout to local API and monitoring.
It uses local/free services and avoids model training unless explicitly run.
The top-level README is the quick command index; this guide explains order and
recovery. Tested local environment: Linux x86_64, Docker, kind, Python 3.10.

## 1. Checkout and retrieve data

Install the supported Python environment from README, Docker, `kubectl`,
`kind`, `envsubst`, DVC and MLflow requirements. Configure `.env` from
`.env.example` without committing secrets. For the author's local RustFS:

```bash
scripts/configure_dvc_remote.sh --start-local-storage
python scripts/ensure_dvc_bucket.py
scripts/configure_dvc_remote.sh
dvc pull
dvc repro validate_data
```

Another computer needs a reachable DVC endpoint and shared credentials, and
enough disk for all three datasets. Never expose local RustFS credentials in
Git or a public endpoint. DVC friends access is not yet demonstrated.

## 2. Evaluate without training

The default DVC graph includes expensive teacher and student stages. To evaluate
the checked-out approved checkpoint only, use single-stage commands:

```bash
dvc repro --single-item evaluate_target
dvc repro --single-item evaluate_ood
```

Each run writes JSON metrics/confusion matrix under `reports/` and an MLflow
evaluation run. These commands require datasets and the existing
`student_model.pth`; they do not retrain. To intentionally train, use the
dedicated commands/configurations in the README and confirm the chosen CPU/GPU
environment first.

## 3. Inspect MLflow and package

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>. The local SQLite store is not shared with
teammates. Verify model sidecar/checkpoint identity and strict size cap before
building:

```bash
python scripts/create_release_metadata.py --validate-only
```

When using the helper in a container, MLflow artifact URIs may be absolute;
mount the checkout at its corresponding host path or configure an accessible
artifact root. Do not create a GHCR release tag unless intentionally ready to
publish; this project has not yet published a versioned image.

## 4. Build and test the local API image

```bash
docker build -t vehicle-classifier-api:local .
docker run -d --name vehicle-classifier-api --network none \
  vehicle-classifier-api:local
docker exec vehicle-classifier-api python /app/scripts/smoke_test_api.py
docker rm -f vehicle-classifier-api
```

For host access during development, run with `-p 8000:8000` instead of
`--network none`, then run `python scripts/smoke_test_api.py` on the host. The
network-disabled smoke runs inside the container and proves the packaged
inference path does not download model weights at runtime.

## 5. Deploy to local kind

Create the disposable cluster once (`kind create cluster --name
vehicle-classifier-phase6`) or use the existing context. Build and load the
image; render Kubernetes configuration with the champion metadata:

```bash
docker build -t vehicle-classifier-api:local .
kind load docker-image vehicle-classifier-api:local \
  --name vehicle-classifier-phase6
export IMAGE_REF=vehicle-classifier-api:local
export MODEL_VERSION=2
export MODEL_SHA256=c80eaf7a45e87e604b2e41af992094425834e46d747315a6fb66a65f7c18e1d4
export RELEASE_GIT_SHA="$(git rev-parse HEAD)"
envsubst '${MODEL_VERSION} ${MODEL_SHA256} ${RELEASE_GIT_SHA}' \
  < k8s/configmap.yaml.tpl | kubectl --context kind-vehicle-classifier-phase6 apply -f -
envsubst '${IMAGE_REF} ${MODEL_VERSION} ${RELEASE_GIT_SHA}' \
  < k8s/deployment.yaml.tpl \
  | sed 's/imagePullPolicy: IfNotPresent/imagePullPolicy: Never/' \
  | kubectl --context kind-vehicle-classifier-phase6 apply -f -
kubectl --context kind-vehicle-classifier-phase6 rollout status \
  deployment/vehicle-classifier-api -n vehicle-classifier-staging --timeout=4m
```

Apply monitoring manifests and wait for both deployments:

```bash
kubectl --context kind-vehicle-classifier-phase6 apply -f k8s/monitoring/prometheus.yaml
kubectl --context kind-vehicle-classifier-phase6 apply -f k8s/monitoring/grafana.yaml
kubectl --context kind-vehicle-classifier-phase6 rollout status deployment/vehicle-classifier-prometheus -n vehicle-classifier-staging --timeout=5m
kubectl --context kind-vehicle-classifier-phase6 rollout status deployment/vehicle-classifier-grafana -n vehicle-classifier-staging --timeout=5m
```

In separate terminals forward API, Grafana and Prometheus ports using the
commands in README. Run `python scripts/k8s_smoke_test.py` against the API
forward, check Prometheus **Status → Targets**, and open the provisioned
Grafana dashboard. Use `scripts/load_test.py` only for a documented controlled
local run; the existing result is synthetic and not a production SLO.

## 6. Failure diagnosis and rollback

```bash
kubectl --context kind-vehicle-classifier-phase6 get pods -n vehicle-classifier-staging
kubectl --context kind-vehicle-classifier-phase6 describe deployment/vehicle-classifier-api -n vehicle-classifier-staging
kubectl --context kind-vehicle-classifier-phase6 logs deployment/vehicle-classifier-api -n vehicle-classifier-staging
kubectl --context kind-vehicle-classifier-phase6 rollout history deployment/vehicle-classifier-api -n vehicle-classifier-staging
kubectl --context kind-vehicle-classifier-phase6 rollout undo deployment/vehicle-classifier-api -n vehicle-classifier-staging
kubectl --context kind-vehicle-classifier-phase6 rollout status deployment/vehicle-classifier-api -n vehicle-classifier-staging --timeout=4m
```

After rollback, repeat readiness and prediction smoke checks before calling the
service restored. Local failed-image rollout and rollback were demonstrated
earlier; Phase 8 successfully redeployed the champion image. Jenkins itself
still needs its runner, kubeconfig credential and a GHCR digest for automated
release/rollback evidence.

## 7. Stop local services

Stop port-forward commands with Ctrl+C. A disposable local cluster can be
removed with `kind delete cluster --name vehicle-classifier-phase6`. That
removes its Prometheus/Grafana history and Kubernetes objects; it does not
remove repository files, Docker images, MLflow data or DVC objects. RustFS is a
separate local service; stop it with the project's storage script when finished.

## Limitations before calling this a portable release

- Local DVC data retrieval in this Phase 8 attempt hit temporary-storage quota;
  use a machine with sufficient free disk for clean-clone verification.
- Local RustFS/MLflow data is not automatically accessible to teammates.
- GitHub Actions branch-protection/release and Jenkins/GHCR deployment require
  account-side setup and have not been evidenced in this checkout.
- No version tag was created or image published.
- Dataset licenses/source provenance are unresolved, and mapped OOD accuracy
  is low; see the data and model cards.
