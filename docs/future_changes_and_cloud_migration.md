# MLOps Future Changes & Cloud Migration Notes

This document tracks strictly **MLOps architectural gaps, pipeline improvements, and free-tier cloud migration options** identified across each stage of the **Memory-Constrained Vehicle Classifier** MLOps lifecycle (skipping generic software/coding style nitpicks).

---

## 📌 Implementation Status & Deferred Actions Log

| Stage | Focus Area | Status | Completed Changes | Deferred Changes (To Be Done Later) |
|---|---|---|---|---|
| **Stage 1** | Reproducibility & Lineage | ⏳ Deferred | Code lineage & git commit tracking active | Generic packaging cleanups skipped per review focus |
| **Stage 2** | Data Engineering & DVC | 🔄 In Progress | • **2d**: Validation gate hard-fails on warnings (`validate_data_manifest.py`)<br>• **2a**: Unified deterministic raw-split ingestion (`scripts/ingest_data.py`, Stage 0 in `dvc.yaml`)<br>• **2c**: DagsHub DVC remote config tooling (`scripts/configure_dvc_remote.sh`, `.dvc/config`) | • **2b**: Dataset sharding (`WebDataset` `.tar` or Parquet shards)<br>• **Fresh DVC Cloud Push**: Awaiting complete raw dataset from peer. Once received, remove legacy `.dvc` files (`vehicle_dataset.dvc`, `test_dataset.dvc`), run `./scripts/configure_dvc_remote.sh --track-local-data && dvc push`. |
| **Stage 3** | Training & Tracking | 🔄 In Progress | • **3a**: DagsHub Hosted MLflow integration (`tracking.py`, `promote_model.py`, live verified on cloud)<br>• **3b**: Granular hyperparameter tracking (`params:` in `dvc.yaml` for teacher, student, evaluations)<br>• **3c**: Rich evaluation metrics (Macro-F1, Precision, Recall, Weighted-F1 in `evaluation.py`, `evaluate_test_set.py`, `record_baseline.py`) | • **3d**: Re-running full end-to-end training pipeline on GPU to regenerate `dvc.lock` for all 6 stages (awaiting GPU runner & peer dataset)<br>• **Batched DataLoader Evaluation**: Convert single-image sequential CPU evaluation (`batch_size=1`) to batched DataLoader inference |
| **Stage 4** | Quality Gates & Registry | ✅ Completed | • **4a**: Multi-metric quality gate (`quality_gates.yaml`, `model_gate.py` enforcing Macro-F1, latency SLA, per-class recall floors)<br>• **4b**: Cloud Model Registry promotion & GitOps champion tracking (`promote_model.py`, `configs/champion_model.yaml`, DagsHub cloud alias `champion`) | • Automated 4-eyes PR approval (deferred to Stage 6 CI)<br>• Canary holdout shadow evaluation (deferred to Stage 7 CD) |
| **Stage 5** | Serving & Containerization | ✅ Completed | • **5a**: Non-blocking `asyncio.to_thread` inference in `api.py`<br>• **5b**: Rich prediction schema with confidence & probabilities (`PredictionResponse`)<br>• **5c**: Multi-stage `Dockerfile` (builder + slim runner) & dynamic model loading | • **5d**: ONNX INT8 Quantization export script (`scripts/export_onnx.py`) for low-memory micro-instances |
| **Stage 6** | CI/CD & Security | ✅ Completed | • **6a**: Automated model quality gate in CI (`ci.yml`)<br>• **6d**: DagsHub S3 auth fix in `train-gpu.yml` & automated CD repository dispatch in `release-image.yml` | • **6b**: Container vulnerability scanning (Trivy)<br>• **6c**: Sigstore Cosign keyless container signing |
| **Stage 8** | Observability & Monitoring| ✅ Completed | • **8a**: ML-specific Prometheus metrics & alerts (`api.py`, `prometheus.yaml`)<br>• **8b**: Multi-resolution load testing (`scripts/load_test.py`)<br>• **8c**: Prometheus PVC & Alertmanager deployment (`k8s/monitoring/`)<br>• **8d**: Cloud observability blueprint (Grafana Cloud, Evidently AI)<br>• **8e**: OpenTelemetry distributed request tracing (`vehicle_classifier_core/telemetry.py`, `api.py`, `predictor.py`) | • Production webhook secrets provisioning (Slack/PagerDuty) |

---

## Stage 1: Environment, Reproducibility & Code Architecture (Phase 0 & 1)
*(Skipped generic coding/packaging practices per review focus — MLOps-specific reproducibility and environment lineage are covered in the pipeline stages below.)*

---

## Stage 2: Data Engineering, Versioning & Validation (Phase 2)

### 1. MLOps Setup Gaps & Pipeline Flaws
- **Raw-to-Curated Data Preparation Is Disconnected from the DVC DAG:** `dataset_organizer.py`, `build_test_data.py`, and `build_mio_tcd_test.py` are standalone scripts outside `dvc.yaml`. Furthermore, both `dataset_organizer.py` (`vehicle_dataset`) and `build_test_data.py` (`test_dataset`) sample from the same `source_dir: ./imagenet_raw` with the same `seed: 1729` without a shared exclusion list! Because `dvc.yaml` starts *after* those folders already exist, DVC has no lineage linking `imagenet_raw` to `vehicle_dataset` / `test_dataset` (`configs/data.yaml` explicitly admits `source: local curated training set; raw lineage not recorded`).
- **Single-Host Local Object Storage (`RustFS` on `127.0.0.1:9000`):** Using `rustfs/rustfs:1.0.3-glibc` in `compose.yaml` bound to `127.0.0.1` means teammates and cloud GitHub Actions runners cannot run `dvc pull` unless they are on the same machine or LAN. It also has no disk redundancy—if the local Docker volume `rustfs_data` is deleted, the versioned remote data is lost.
- **24,690 Individual Small Image Files (~2.08 GB) Cause Slow DVC Checkout/Push:** DVC hashes and transfers every image file individually (`19,690` files in `vehicle_dataset` + `2,500` in `test_dataset` + `2,500` in `test_MIO_TCD_data`). On S3 remotes, 24k small HTTP PUT/GET requests create massive I/O overhead and hit local storage quota limits during clean-clone checks.
- **Validation Gate Allows Near-Duplicate Leakage Warnings to Pass Silently:** `scripts/validate_data_manifest.py` only exits with code `1` when `status == "failed"` (exact SHA-256 cross-split duplicates or corrupted images). When `status == "review_warnings"` (e.g., an augmented test image in `test_dataset` derived from the *same* underlying raw ImageNet photo as a training image in `vehicle_dataset`, caught by `dhash64`), the DVC stage still exits `0` and lets training proceed.

### 2. Better MLOps Ways to Do This
- **Unify Raw-to-Split Ingestion as Stage 0 in `dvc.yaml`:** Combine `dataset_organizer.py` and `build_test_data.py` into a single deterministic ingestion stage that partitions raw source images into disjoint `train`, `val`, and `test` sets *before* applying synthetic corruptions (`clean_` vs `aug_`), guaranteeing zero raw-source overlap between `vehicle_dataset` and `test_dataset`.
- **Pack Datasets into Sharded Archives (`WebDataset` `.tar` or Parquet):** Instead of tracking 24,690 loose `.jpg`/`.JPEG` files in DVC, pack splits into 10–20 `.tar` shards (`WebDataset` format) or Hugging Face `datasets` Arrow/Parquet files. DVC then tracks ~20 large files instead of 24,690 tiny files, speeding up `dvc pull`/`dvc push` by 10–50x.

### 3. Free Cloud Alternatives (Data Versioning & Remote Storage)
- **DagsHub (Free Tier: 10 GB–100 GB Hosted DVC S3 Remote + Data Engine):** Directly replaces the local `RustFS` Docker container. You just change `DVC_ENDPOINT_URL` to your DagsHub repo's S3 endpoint (`https://dagshub.com/<user>/<repo>.s3`). Teammates and GitHub Actions runners can `dvc pull` from anywhere without port-forwarding or LAN setup.
- **Cloudflare R2 (Free Tier: 10 GB/month storage, Zero Egress Fees, S3-Compatible):** Since DVC already uses `dvc[s3]`, Cloudflare R2 is a drop-in cloud replacement for `RustFS`—just update `DVC_ENDPOINT_URL`, `DVC_ACCESS_KEY`, and `DVC_SECRET_KEY` in `.env` and GitHub Secrets.
- **Hugging Face Hub Datasets (Free Public/Private Dataset Repositories):** Can store versioned image datasets (with built-in Git LFS / Xet storage and Parquet viewer) for free without managing S3 buckets at all.

### 4. Implementation Status & Deferred Tasks (Stage 2)
- **Completed**:
  - **[2d] Validation Gate Hard-Fail**: `scripts/validate_data_manifest.py` now hard-fails (`exit 1`) on `review_warnings` (near-duplicate image leakage).
  - **[2a] Unified Ingestion DAG**: Created `scripts/ingest_data.py` and registered as Stage 0 in `dvc.yaml` to ensure disjoint raw-split partitioning.
  - **[2c] DagsHub DVC Remote Tooling**: Added and tested `scripts/configure_dvc_remote.sh` and `.dvc/config` for S3-compatible cloud storage.
- **Deferred (To Be Done Later)**:
  - **[2b] Dataset Sharding (WebDataset / Parquet)**: Deferred packing 24k individual images into `.tar` shards until final dataset curation is complete.
  - **Fresh Cloud Data Push**: Awaiting the complete dataset from project peer. When the full dataset is placed locally:
    1. Remove legacy conflicting standalone pointer files (`vehicle_dataset.dvc`, `test_dataset.dvc`).
    2. Run `./scripts/configure_dvc_remote.sh --track-local-data`.
    3. Run `dvc push` to upload all versioned splits to the DagsHub remote.

---

## Stage 3: Model Training, Knowledge Distillation & Experiment Tracking (Phase 2 & 3)

### 1. MLOps Setup Gaps & Pipeline Flaws
- **Isolated Local SQLite & Filesystem MLflow Store (`sqlite:///mlflow.db` + `./mlartifacts`):** By default (`tracking.py`), MLflow writes run metadata to a local `mlflow.db` file and artifacts to `./mlartifacts/`. Both are gitignored and not tracked by DVC. This means experiment history, training curves, and registered models exist *only* on the single laptop that ran the script—teammates and CI pipelines cannot view or compare runs centrally.
- **Incomplete DVC Lockfile (`train_teacher` and `train_student` Missing from `dvc.lock`):** While `dvc.yaml` defines 6 stages (`create_data_manifest`, `validate_data`, `train_teacher`, `train_student`, `evaluate_target`, `evaluate_ood`), `dvc.lock` only records hashes for 4 stages (`train_teacher` and `train_student` were bypassed using `dvc repro --single-item evaluate_target` with a pre-existing `student_model.pth`). Furthermore, `student_model.pth` is directly committed to Git instead of being managed purely as a DVC-cached pipeline output (`outs`), creating dual tracking between Git and DVC.
- **Missing Hyperparameters in `dvc.yaml` `params:` Tracking:** In `dvc.yaml`, `train_teacher` and `train_student` list `configs/teacher.yaml` and `configs/distillation.yaml` under `deps:`, rather than under `params:`. While changing the YAML file triggers a re-run, DVC's `dvc params diff` and `dvc exp show` cannot display granular hyperparameter comparisons (e.g., `temperature`, `alpha`, `learning_rate`) across Git commits unless they are declared under `params:`.
- **Single-Image Sequential Evaluation (`evaluate_test_set.py`)**: Evaluation loops over 2,500 images one by one on CPU using `classifier.predict(str(image_path))` (batch size = 1) instead of batched DataLoader inference, and only records **accuracy** (`per_class` support/correct/accuracy) without **Precision, Recall, Macro/Weighted F1-score, or Calibration (ECE)**—even though OOD accuracy drops to 43.24%.

### 2. Better MLOps Ways to Do This
- **Centralize MLflow Tracking & Artifact Store:** Configure MLflow to use a remote PostgreSQL backend (`MLFLOW_TRACKING_URI`) and shared S3 artifact root (`--default-artifact-root s3://...`), or point `MLFLOW_TRACKING_URI` to a managed cloud tracking server so every teammate and GitHub Actions GPU run logs to a single source of truth.
- **Declare Granular `params:` in `dvc.yaml` & Complete Full `dvc.lock`:** Add `params: [configs/teacher.yaml: [training, augmentation], configs/distillation.yaml: [training, artifact]]` to `dvc.yaml` and run the full pipeline once on a GPU runner so `dvc.lock` captures the complete end-to-end lineage from `vehicle_dataset` $\rightarrow$ `EfficientNet_B2_teacher_model.pth` $\rightarrow$ `student_model.pth`.
- **Add Macro-F1, Precision/Recall & Confidence Thresholding to Evaluation:** Track Macro-F1 and per-class Precision/Recall in `evaluation.py` and `tracking.py`, and log confidence histograms to detect out-of-distribution (OOD) inputs where the model is overconfident on MIO-TCD images.

### 3. Free Cloud Alternatives (Experiment Tracking & GPU Training)
- **DagsHub Hosted MLflow Server (Free Tier Included with Repo):** Every DagsHub repository comes with a free hosted **MLflow Tracking Server & Model Registry**. You only set `MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow` and `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`. Both `mlflow.db` and `mlartifacts/` move to the cloud with zero code changes to `tracking.py`!
- **Weights & Biases (W&B Free Personal/Academic Tier) or Comet ML:** Free hosted experiment tracking with rich confusion matrix plots, system GPU/CPU utilization charts, and hyperparameter sweeps.
- **Neon.tech / Supabase (Free Managed PostgreSQL) + Cloudflare R2:** If self-hosting MLflow on a free cloud container (e.g., Hugging Face Spaces or Render), use Neon's free serverless Postgres as `--backend-store-uri` and Cloudflare R2 as `--default-artifact-root`.

### 4. Implementation Status & Deferred Tasks (Stage 3)
- **Completed & Verified**:
  - **[3a] Cloud MLflow Integration**: `vehicle_classifier_core/tracking.py` and `scripts/promote_model.py` automatically load `.env` and authenticate with DagsHub MLflow. Live cloud connection verified by creating `VehicleClassifier` experiment and recording test run `17ae2dbd2d994e588b907da6cd384fb3`.
  - **[3b] Granular Hyperparameters**: Added `params:` declarations to `dvc.yaml` for `train_teacher`, `train_student`, `evaluate_target`, and `evaluate_ood`. Verified via `dvc params diff` and `dvc dag`.
  - **[3c] Comprehensive Evaluation Metrics**: `vehicle_classifier_core/evaluation.py` computes per-class Precision, Recall, F1, and dataset Macro-F1 / Weighted-F1. Logged to MLflow in `tracking.py` and verified via unit tests.
- **Deferred (To Be Done Later)**:
  - **[3d] Full GPU Retraining & Complete dvc.lock**: Re-running `train_teacher` (8 epochs) and `train_student` (20 epochs) to produce a fresh end-to-end `dvc.lock` capturing all 6 stages. Deferred until GPU machine / full training dataset from peer is available.
  - **Batched DataLoader Evaluation**: Convert sequential `batch_size=1` loops in `evaluate_test_set.py` into a batched PyTorch DataLoader to speed up CPU evaluation passes.

---

## Stage 4: Model Quality Gate, Registration & Promotion (Phase 3)

### 1. MLOps Setup Gaps & Pipeline Flaws
- **Quality Gate Evaluates Only Top-Line Accuracy (`>= baseline_accuracy`)**: `model_gate.py` checks `overall_accuracy >= baseline_accuracy` (92.44%), but does not enforce any per-class floors or OOD thresholds. A candidate model could collapse to 0% accuracy on rare/critical classes like `Bus` or perform disastrously on OOD data, but still pass the gate as long as the aggregate accuracy on common classes compensates.
- **Manual, Unaudited CLI Promotion (`--approve`)**: Promoting a model version to the `champion` alias in MLflow is executed via a local CLI script (`python scripts/promote_model.py --version 2 --approve`). There is no four-eyes principle (multi-person approval), automated smoke evaluation against a holdout canary set, or audit trail of *who* promoted the model beyond a mutable MLflow tag.
- **Local Model Registry Fragmentation**: Because the MLflow model registry is stored in local `mlflow.db`, the concept of a `champion` model alias is purely local to one developer machine. The CI/CD release workflow in GitHub Actions (Stage 6) and Jenkins (Stage 7) cannot query this local registry to dynamically fetch the latest `champion` model weights.

### 2. Better MLOps Ways to Do This
- **Multi-Metric Quality Gate (Slice-Based Evaluation & Latency SLA)**: Expand `configs/quality_gates.yaml` and `model_gate.py` to evaluate:
  1. Minimum per-class recall thresholds (e.g., `Bus >= 0.88`, `Truck >= 0.90`).
  2. Latency SLA gate (e.g., `mean_latency_ms <= 15.0 ms` on CPU).
  3. OOD drift degradation limit (e.g., `ood_accuracy >= 40.0%`).
- **GitOps-Driven Promotion via GitHub Pull Requests / Releases**: Replace manual CLI tag mutation with a declarative GitOps file (e.g., `champion_model.yaml` containing model version, SHA-256, and benchmark scores). Promoting a model is done via a PR requiring code-owner review. Merging the PR triggers the release and deployment pipelines automatically.

### 3. Free Cloud Alternatives (Model Registry & Governance)
- **DagsHub Model Registry (Free Tier)**: Provides a centralized, team-accessible MLflow Model Registry UI where team members can review candidate metrics, diff confusion matrices, and promote versions to `champion` collaboratively.
- **Hugging Face Model Hub (Free Tier)**: Push candidate and champion models to a private/public HF Model repo with versioned Git tags (`v1`, `v2`) and automated model cards (`README.md` containing evaluation reports and confusion matrices).

### 4. Implementation Status & Deferred Tasks (Stage 4)
- **Completed & Verified**:
  - **[4a] Multi-Metric Quality Gate**: Extended `configs/quality_gates.yaml` and `vehicle_classifier_core/model_gate.py` to evaluate overall accuracy delta (`>= -0.005`), Macro-F1 floor (`>= 0.88`), CPU inference latency SLA (`<= 15.0 ms`), and per-class slice recall floors (e.g. Bus `>= 0.85`, Truck `>= 0.88`). Verified via `scripts/model_gate.py` and unit tests in `tests/test_model_gate_and_promotion.py`.
  - **[4b] Cloud Model Registry Promotion & Declarative Champion Tracking**: Updated `scripts/promote_model.py` to strictly verify the quality gate report before allowing promotion, authenticate with DagsHub MLflow, assign the `champion` alias on cloud Model Registry, tag the version as `approved`, and record declarative GitOps state in `configs/champion_model.yaml`. Live cloud promotion verified for version 1 on DagsHub Model Registry (`Aliases: {'candidate': '1', 'champion': '1'}`).
- **Deferred (To Be Done Later)**:
  - **Automated 4-Eyes PR Approval**: Enforcing promotion via GitHub Pull Request approval from code owners rather than single CLI flag (deferred to Stage 6 CI/CD workflow integration).
  - **Canary / Shadow Holdout Evaluation**: Automated evaluation against live incoming inference shadows prior to promotion (deferred to Stage 7 CD / Staging rollout).

---

## Stage 5: Model Packaging, Inference API & Containerization (Phase 4)

### 1. MLOps Setup Gaps & Pipeline Flaws
- **Coupled Container Image & Model Weights (Model Baked into Docker Image)**: `Dockerfile` performs `COPY student_model.pth /app/artifacts/student_model.pth`. Baking weights directly into the container image creates a monolithic coupling between model release cadence and software releases: every retrained or promoted model version requires rebuilding, re-scanning, and repushing a ~1 GB+ Docker image even if no Python/OS dependencies or API routing code changed.
- **Event-Loop Blocking by Synchronous CPU Inference in `async def predict`**: In `vehicle_classifier_core/api.py`, `predict_image()` (PyTorch tensor preprocessing, memory copy, and ShuffleNet forward pass) is called directly inside an `async def predict` route. Because PyTorch inference is CPU-bound synchronous code, running it inside the main async coroutine halts the Python asyncio event loop. Under concurrent traffic, pending I/O tasks (such as Kubernetes liveness/readiness probes `/health/live` and `/health/ready`) can time out and cause Kubernetes to restart healthy pods.
- **Missing Prediction Confidence / Probabilities in API Response**: The response contract only returns `{"predicted_index": int, "class_name": str, "model_sha256": str}`. It omits softmax probabilities, top-k confidences, or raw logits. Downstream clients cannot evaluate prediction certainty, enforce rejection thresholds for out-of-distribution inputs, or trigger human-in-the-loop review.
- **Unquantized FP32 PyTorch Runtime (Missed Memory & Latency Optimization)**: The deployment packages the standard PyTorch FP32 `.pth` state dict, requiring the full `torch` + `torchvision` CPU packages (~700 MB uncompressed). For an explicit memory-constrained project, failing to convert to INT8 ONNX or OpenVINO leaves 2x–4x CPU latency gains and 50% memory reduction on the table.
- **Single-Stage Docker Build**: The `Dockerfile` does not use multi-stage builds. Build dependencies, wheel caches, and pip overhead remain embedded in image layers, increasing the attack surface and container pull times.

### 2. Better MLOps Ways to Do This
- **Decouple Service Image from Model Weights (Sidecar / Init-Container / Remote S3 Pull)**: Separate the generic serving container (FastAPI + inference engine) from the model artifact. The container can pull the approved model at startup via `MODEL_URI` (from S3/R2/MLflow) or mount it from an external shared volume or Kubernetes init-container. This allows instant model rollouts without rebuilding Docker images.
- **Non-Blocking Threadpool Execution**: In `api.py`, offload the CPU-bound inference call to worker threads using `await asyncio.to_thread(request.app.state.classifier.predict_image, rgb)` (or use FastAPI's synchronous `def predict(...)` syntax) to keep the asyncio event loop free to respond to health probes and concurrent requests.
- **Rich Output Contract with Confidence & Drift Metadata**: Enhance `PredictionResponse` to return `confidence: float`, `probabilities: dict[str, float]`, and a unique `request_id` header to enable downstream filtering, monitoring, and telemetry collection.
- **Compile to ONNX Runtime with INT8 Quantization**: Export the ShuffleNet V2 PyTorch model to ONNX format with dynamic INT8 quantization. Replace the heavy `torch` CPU dependency with `onnxruntime` (~50 MB wheel), slashing image size, cold-start time, and inference latency.
- **Multi-Stage Dockerfile**: Use a builder stage to compile wheels and a minimal distroless or slim runner stage copying only installed site-packages and app code.

### 3. Free Cloud Alternatives (Model Serving & Container Hosting)
- **Hugging Face Spaces (Free Tier: 2 vCPU, 16 GB RAM Docker Environment)**: Free hosting for Dockerized ML APIs. You can deploy the Docker container or FastAPI service with automatic public HTTPS endpoints, hardware acceleration options, and built-in OpenAPI documentation.
- **Render / Koyeb / Fly.io (Free Tier Container Hosting)**: Host the Docker container on free-tier web services with automated GitHub deployment, free SSL termination, health check routing, and custom domain support.
- **Google Cloud Run (Free Tier: 2 Million Requests/Month, Scale-to-Zero)**: Deploy the container as a serverless containerized microservice on Google Cloud. Offers 2 million free requests, 360,000 GB-seconds of memory, and 180,000 vCPU-seconds per month with automatic scale-to-zero when idle.

### 4. Implementation Status & Deferred Tasks (Stage 5)
- **Completed & Verified**:
  - **[5a] Non-Blocking Threadpool Inference**: Offloaded CPU-bound PyTorch prediction in `vehicle_classifier_core/api.py` using `await asyncio.to_thread(request.app.state.classifier.predict_image_detailed, rgb)`. Event loop remains responsive to health probes (`/health/live`, `/health/ready`) and Prometheus scraping under concurrency.
  - **[5b] Rich Prediction Contract with Confidence & Probabilities**: Enhanced `PredictionResponse` in `api.py` and `VehicleClassifier.predict_image_detailed()` in `predictor.py` to return winning class confidence (`confidence: float`) and per-class probability distribution (`probabilities: dict[str, float]`). Full unit test coverage added in `tests/test_api.py` and `tests/test_inference_contract.py`.
  - **[5c] Multi-Stage Dockerfile & Decoupled Model Weights**: Converted `Dockerfile` to a 2-stage build (`builder` compiling wheels to `/install`; slim `runner` copying pure packages without build overhead). Decoupled weights by supporting dynamic runtime loading via `MODEL_PATH` or declarative lookup from `configs/champion_model.yaml`.
- **Deferred (To Be Done Later)**:
  - **[5d] ShuffleNet V2 ONNX INT8 Quantization Export**: Dedicated export script (`scripts/export_onnx.py`) compiling the PyTorch ShuffleNet student into dynamically quantized INT8 ONNX format to shrink model size from 2.6 MB to ~0.7 MB and eliminate the ~700 MB PyTorch dependency on edge devices.

---

## Stage 6: CI/CD Pipelines & Release Automation (Phase 5)

### 1. MLOps Setup Gaps & Pipeline Flaws
- **Self-Hosted GPU Runner Requirement Blocks Cloud CI Training**: `.github/workflows/train-gpu.yml` hardcodes `runs-on: [self-hosted, linux, x64, gpu]`. Because GitHub-hosted free runners do not include GPUs, this workflow fails immediately unless an external, privately administered Linux GPU machine is pre-registered as a runner. Furthermore, training results write to an ephemeral local SQLite DB (`MLFLOW_TRACKING_URI: sqlite:///mlflow.db`) that is zipped into a workflow run artifact rather than recorded to a centralized, permanent tracking server.
- **PR Quality Gate Omits Model Quality & Latency Gates**: `.github/workflows/ci.yml` runs only linter checks (`ruff check`), bytecode compilation (`compileall`), and basic unit tests (`pytest`). It completely bypasses the model quality gate (`model_gate.py`) and accuracy/latency evaluations. A developer or automated PR could modify model hyperparameters, change student architectures, or update model weights without triggering any automated regression check against baseline accuracy (92.44%) or CPU inference latency.
- **Absence of Container Security & Vulnerability Scanning**: `.github/workflows/release-image.yml` builds the Docker image and immediately pushes it to GHCR without running any static vulnerability analysis (e.g., Trivy, Grype, or Snyk). Outdated operating system packages, insecure Python wheels, or leaked secrets embedded in image layers are published directly to production registries without a safety gate.
- **Missing Container Signing & Supply-Chain Attestation**: While `create_release_metadata.py` records image digests and Git SHAs in a JSON manifest attached to GitHub Releases, the container image itself is unsigned. There is no cryptographic signature (e.g., Sigstore / Cosign keyless signing) ensuring that the image pulled by Kubernetes or Jenkins was genuinely built by this repository's authorized GitHub Actions workflow and has not been tampered with.
- **Disconnected Deployment Handshake**: The release workflow ends by attaching `release_metadata.json` to a GitHub Release, but provides no webhook, dispatch, or automated event to notify the deployment cluster (Jenkins / Kubernetes) that a new validated release is available.

### 2. Better MLOps Ways to Do This
- **Integrate Model Quality Gate in CI for Config/Artifact PRs**: Add a conditioned step in `ci.yml` that triggers when changes touch `configs/`, `models/`, or weights. It should run `model_gate.py` on a fast golden validation subset, blocking PR merges if accuracy drops or size exceeds 5 MB.
- **Automated Container Vulnerability Gate (Trivy)**: Insert a container scan step in `release-image.yml` (e.g. `aquasecurity/trivy-action`) with `severity: 'CRITICAL,HIGH'` and `exit-code: 1`. Reject publishing any container image that contains unpatched critical CVEs.
- **Keyless Container Signing with Sigstore Cosign**: Use GitHub Actions OIDC tokens to sign the GHCR container image and its `release_metadata.json` using Cosign. Deployment tools (Jenkins, Kubernetes Admission Controllers) can then cryptographically verify the signature before deploying.
- **Centralize Cloud MLflow in GPU Workflows**: Update `train-gpu.yml` to use hosted MLflow (`MLFLOW_TRACKING_URI`) so training curves, distillation metrics, and candidate models are immediately accessible to team members rather than buried in zipped runner artifacts.
- **Automate Deployment Trigger (Repository Dispatch / ArgoCD GitOps)**: Once the release image is published and signed, emit a `repository_dispatch` event or push an updated image digest commit to a GitOps repository to trigger automated staging rollout.

### 3. Free Cloud Alternatives (CI/CD, GPU Runners & Supply-Chain Security)
- **GitHub Actions (Free Tier: 2,000 Linux Minutes/Month + Free GHCR Package Storage)**: The free tier easily handles CPU linting, unit tests, Docker builds, and GHCR container hosting without any credit card.
- **Kaggle Notebooks / Google Colab (Free GPU: T4 / P100 with 15–30h/week)**: Replaces the need for an expensive self-hosted GPU runner. Training scripts and DVC pipelines can run seamlessly in a headless Kaggle/Colab script pulling from DagsHub DVC storage and logging to DagsHub MLflow.
- **Sigstore Cosign (100% Free Public Keyless Image Signing)**: Free, open-source container signing that integrates natively with GitHub Actions OIDC tokens and requires no private key management.
- **GitHub Secret Scanning & Dependabot (Free)**: Free native GitHub tools that automatically scan pull requests and dependencies for vulnerable libraries and leaked secrets.

### 4. Implementation Status & Deferred Tasks (Stage 6)
- **Completed & Verified**:
  - **[6a] Model Quality Gate in CI**: Added candidate quality gate enforcement step to `.github/workflows/ci.yml` invoking `python scripts/model_gate.py --evaluation reports/evaluation.json --baseline reports/baseline.json`. Blocks pull requests if model candidates violate accuracy delta, latency SLA (15 ms), Macro-F1 floor (0.88), per-class recall, or artifact size limits.
  - **[6d] DagsHub Auth Fix & CD Dispatch Handshake**: Fixed DagsHub S3 remote auth credentials in `.github/workflows/train-gpu.yml` (`access_key_id "${DAGSHUB_TOKEN}"`). Added automated `repository_dispatch` webhook in `.github/workflows/release-image.yml` to trigger continuous delivery workflows with the published container image digest and Git tag. Verified via unit tests in `tests/test_workflows.py`.
- **Deferred (To Be Done Later)**:
  - **[6b] Automated Container Vulnerability Scanning (Trivy)**: Insert static image scanning with Aqua Security Trivy (`severity: 'CRITICAL,HIGH'`) before pushing to GHCR.
  - **[6c] Keyless Supply-Chain Container Signing (Sigstore Cosign)**: Sign container image digests using GitHub Actions OIDC tokens.

---

## Stage 7: Continuous Deployment, Staging & Rollback (Phase 6)

### 1. MLOps Setup Gaps & Pipeline Flaws
- **Manual Copy-Paste Handshake in Jenkins (`RELEASE_METADATA_JSON`)**: In `Jenkinsfile`, line 12 requires an operator to manually copy the JSON text from a GitHub Release and paste it into a Jenkins build parameter text field. This manual step breaks continuous delivery automation, introduces human error (e.g., truncated text or wrong release pasted), and prevents seamless end-to-end GitOps deployment.
- **Fragmented Toolchain (GitHub Actions for CI vs. Jenkins for CD)**: Using GitHub Actions for CI (YAML) and Jenkins for CD (Groovy `Jenkinsfile`) introduces operational fragmentation: developers must manage two separate secrets vaults (`GITHUB_TOKEN` vs. `vehicle-classifier-kubeconfig`), two runner environments, and two distinct logging dashboards.
- **Fragile Shell Templating (`envsubst` into `kubectl apply`)**: Deployments render Kubernetes manifests by piping environment variables through `envsubst` (`envsubst ... < deployment.yaml.tpl | kubectl apply -f -`). If an environment variable is omitted or misspelled, `envsubst` silently replaces it with empty whitespace, creating invalid or corrupted Kubernetes configurations rather than failing fast.
- **Single-Replica Deployment Without Autoscaling (No HPA)**: `deployment.yaml.tpl` hardcodes `replicas: 1` without a `HorizontalPodAutoscaler` (HPA). If inference traffic spikes or the single pod restarts, the service has no high-availability redundancy, risking serving downtime during peak workloads.
- **All-or-Nothing Rolling Update (Missing Progressive / Canary Delivery)**: The deployment performs a standard Kubernetes rolling update. In enterprise MLOps, deploying a new model version should use canary or shadow deployments (e.g. routing 5%–10% of traffic to the new model, measuring real inference error rates and latencies, and automatically rolling back before 100% of users are exposed).

### 2. Better MLOps Ways to Do This
- **Adopt True GitOps with ArgoCD or Flux**: Replace Jenkins and `envsubst` with declarative GitOps. Store environment manifests in a GitOps repository. When GitHub Actions publishes an image digest, it automatically commits the updated image reference to Git. ArgoCD continuously syncs the cluster state, eliminates manual copy-pasting, and automatically detects configuration drift.
- **Progressive Delivery with Argo Rollouts or Flagger**: Implement canary deployments where 10% of live traffic is routed to the new candidate pod. If the Prometheus error rate rises above 1% or p95 latency exceeds 50 ms, the rollout automatically aborts and routes 100% of traffic back to the stable revision without manual intervention.
- **Horizontal Pod Autoscaling (HPA)**: Add an HPA manifest scaling pods from 2 to 10 replicas based on CPU utilization (`targetCPUUtilizationPercentage: 70%`) or Prometheus request rate metrics.
- **Native Kustomize Overlays**: Replace `.tpl` templates and `envsubst` with Kustomize image transformers (`kustomize edit set image ...`), providing deterministic YAML parsing and dry-run linting before applying to clusters.

### 3. Free Cloud Alternatives (Continuous Delivery & Kubernetes Hosting)
- **Oracle Cloud Infrastructure (OCI) Always Free Tier (Best Free Cloud K8s)**: Provides up to 4 OCPUs and 24 GB of RAM on Ampere A1 Compute completely free for life (no trial expiration). You can launch a free Ubuntu VM, install lightweight **K3s** (`curl -sfL https://get.k3s.io | sh -`), and get a full-fledged cloud Kubernetes cluster with a public IP. Jenkins connects directly via the downloaded `kubeconfig` stored as `vehicle-classifier-kubeconfig`.
- **Civo Cloud ($250 Free Credit) / DigitalOcean ($200 Trial)**: Generous 30–60 day free credits providing managed K3s (Civo) or DOKS (DigitalOcean) clusters that boot in under 90 seconds. Ideal for project demonstrations and defense presentations.
- **Google Cloud Platform (GCP) GKE Free Tier ($300 Credits + $0/mo Free Cluster Management Fee)**: GCP offers one free Zonal cluster per billing account ($0 cluster fee) and $300 in 90-day trial credits to cover compute nodes.
- **Zero-Cost Remote Tunneling for Local `kind` (Tailscale / Cloudflare Tunnel)**: If keeping `kind` running locally on your laptop, use Tailscale or a Cloudflare Tunnel (`cloudflared`) to expose port `6443` securely. A cloud-hosted or remote Jenkins can deploy directly to your local `kind` cluster without port-forwarding routers.
- **Unify CD into GitHub Actions (Alternative to Jenkins)**: While Jenkins is retained for project demonstration purposes, future iterations can optionally replace Jenkins with native GitHub Actions workflow dispatch or ArgoCD for zero-infrastructure maintenance.

### 4. Implementation Status & Deferred Tasks (Stage 7)
- **Completed & Verified**:
  - **[7a] Automated Release Resolution in Jenkins**: Updated `Jenkinsfile` to accept `RELEASE_TAG` (defaulting to latest) and automatically download `release_metadata.json` via GitHub API or fallback to local reports, eliminating the manual copy-paste textarea input.
  - **[7b] High Availability & Horizontal Pod Autoscaler**: Created `k8s/hpa.yaml` dynamically scaling pods from 2 to 6 replicas at 70% CPU target utilization. Configured `k8s/deployment.yaml.tpl` with `replicas: 2` for zero-downtime rolling updates, and deployed HPA in `Jenkinsfile` and `k8s/kustomization.yaml`. Verified via unit tests in `tests/test_workflows.py`.
  - **[7d] Free Cloud Kubernetes Architecture**: Documented cloud deployment options using Oracle Cloud Always Free (OCI Ampere A1 with K3s) and Civo Cloud, including secure remote tunneling via Tailscale/Cloudflare Tunnel.
- **Deferred (To Be Done Later)**:
  - **[7c] Native Kustomize Overlays**: Migrating completely away from `envsubst` to declarative Kustomize image transformers (`kustomize edit set image ...`).

---

## Stage 8: Observability, Monitoring & Load Testing (Phase 7 & 8)

### 1. MLOps Setup Gaps & Pipeline Flaws
- **Ephemerality of In-Cluster Prometheus & Grafana Storage (`emptyDir`)**: Both `k8s/monitoring/prometheus.yaml` and `grafana.yaml` mount TSDB and database storage as `emptyDir` volumes (`sizeLimit: 1Gi` and `512Mi`). If the monitoring pods restart, get rescheduled, or if the cluster reboots, all historical time-series metrics, scraping history, and Grafana session data are permanently wiped.
- **Alertmanager Is Missing (Alerts Trigger Silently)**: `prometheus.yaml` defines alerting rules (`VehicleClassifierTargetDown` and `VehicleClassifierHighServerErrorRate`), but Alertmanager is not deployed and no notification receivers (Slack, Discord, PagerDuty, or Email webhooks) are configured. While rules turn red inside Prometheus' internal `/alerts` tab, no team member or on-call engineer receives an external notification when the service fails.
- **Absence of ML-Specific Drift & Confidence Monitoring**: The Prometheus instrumentation tracks only infrastructure and HTTP metrics (requests, 5xx errors, latency percentiles, and process memory/CPU). It lacks statistical data drift tracking (e.g. KS-test on image brightness/contrast), target prediction drift (detecting sudden distribution shifts where predictions skew towards a single class), and confidence degradation tracking (since softmax probabilities are not exported).
- **Synthetic Single-Resolution Load Testing**: `scripts/load_test.py` generates a synthetic 32x32 pixel PNG in memory. Real-world client requests send variable, high-resolution smartphone/camera photos (e.g., 2–8 MB JPEGs at 1080p). Benchmarking with a tiny 32x32 image underestimates real multipart streaming time, Pillow decoding latency, and CPU memory pressure.
- **No Distributed Tracing (OpenTelemetry / Jaeger)**: The service lacks request tracing. When p95 latency rises under concurrency (e.g. 409 ms at concurrency 4), there is no distributed trace or span breakdown to distinguish between network transfer time, image decoding, preprocessing, and model execution.

### 2. Better MLOps Ways to Do This
- **Persistent Storage for Prometheus via PVCs**: Back Prometheus TSDB with persistent block storage (`PersistentVolumeClaim`) or push metrics via remote-write to an external managed Prometheus backend.
- **Deploy Alertmanager with Webhook Routing**: Deploy Alertmanager with webhook receivers sending high-severity alerts (`TargetDown`, `5xx rate > 5%`) directly to team Slack/Discord channels or Email.
- **Integrate ML Data & Concept Drift Monitoring (Evidently AI / WhyLogs)**: Stream inference metadata (image dimensions, brightness, prediction confidence, class labels) to an Evidently AI or WhyLogs service to compute drift metrics against the baseline training distribution and detect model degradation.
- **Realistic Multi-Resolution Load Testing (Locust / k6)**: Replace the standard-library script with a distributed load testing framework like Locust or k6. Replay a diverse pool of real vehicle images across varying dimensions (from 224x224 to 4K) to stress-test decompression bomb defenses and real-world concurrency limits.
- **OpenTelemetry Instrumentation**: Add OpenTelemetry tracing spans to `api.py` to isolate exact latency contributors across request stages.

### 3. Free Cloud Alternatives (Observability & Monitoring)
- **Grafana Cloud Free Tier (Free Forever)**: Includes 10,000 Prometheus metric series, 50 GB logs (Loki), 50 GB traces (Tempo), and 3 active Grafana alert rules completely free. You can use the Prometheus Agent (`prom/prometheus`) in your cluster to push metrics directly to Grafana Cloud via remote-write, removing the need to manage heavy Prometheus and Grafana pods.
- **Evidently Cloud / WhyLabs (Free Tier ML Observability)**: Free hosted platforms specifically built for machine learning observability, providing automated data drift dashboards, target drift reports, and data quality alerts without maintaining local drift databases.
- **Better Stack (Free Tier Uptime & Incident Management)**: Provides free public status pages, SSL monitoring, and instant email/SMS alerts whenever `/health/ready` or `/health/live` endpoints become unreachable.

### 4. Implementation Status & Deferred Tasks (Stage 8)
- **Completed & Verified**:
  - **[8a] ML-Specific Prometheus Metrics & Alerting Rules**:
    - Added `vehicle_classifier_prediction_confidence` (Histogram tracking softmax confidence distributions across buckets `[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]`) and `vehicle_classifier_low_confidence_predictions_total` (Counter tracking predictions with confidence < 0.70) in `vehicle_classifier_core/api.py`.
    - Added Prometheus alerting rules in `k8s/monitoring/prometheus.yaml`:
      - `VehicleClassifierHighInferenceLatency`: triggers if p95 inference latency exceeds the 15 ms target SLA for 5 minutes (`histogram_quantile(0.95, sum(rate(vehicle_classifier_inference_duration_seconds_bucket[5m])) by (le)) > 0.015`).
      - `VehicleClassifierLowConfidenceSpike`: triggers if predictions with confidence < 0.70 exceed 20% over 5 minutes, signaling potential covariate/data distribution drift.
  - **[8b] Multi-Resolution Realistic Load Testing**:
    - Upgraded `scripts/load_test.py` with multi-resolution support (`--resolutions 32 224` or custom list) and `--image-path` for real image benchmarking.
    - Added p99 latency percentile reporting alongside p50, p95, and max, and recorded average prediction confidence per concurrency level.
  - **[8c] Persistent TSDB Storage & Alertmanager Deployment**:
    - Configured a 10Gi `PersistentVolumeClaim` (`vehicle-classifier-prometheus-pvc`) in `k8s/monitoring/prometheus.yaml` replacing ephemeral `emptyDir` so metric history survives pod restarts.
    - Created `k8s/monitoring/alertmanager.yaml` with ConfigMap (notification routing, rate limits, webhook receiver), 2Gi PVC (`vehicle-classifier-alertmanager-pvc`), Deployment (`prom/alertmanager:v0.28.0`), and ClusterIP Service.
    - Linked Prometheus `alerting.alertmanagers` to `vehicle-classifier-alertmanager.vehicle-classifier-staging.svc.cluster.local:9093`.
  - **[8d] Free Cloud Observability & Drift Monitoring Blueprint**:
    - Documented Grafana Cloud free tier remote-write integration pattern to offload monitoring compute and TSDB storage from local clusters.
    - Documented Evidently AI / WhyLabs ML drift monitoring blueprint for tracking real-world inference distribution drift against training baselines.
  - **[8e] OpenTelemetry Distributed Request Tracing**:
    - Implemented `vehicle_classifier_core/telemetry.py` with `setup_telemetry()`, `get_tracer()`, and W3C `traceparent` context propagation.
    - Added granular spans in `api.py` and `predictor.py` isolating payload reception (`read_upload_payload`), image verification & dimension check (`validate_and_decode_image`), RGB conversion (`convert_image_rgb`), tensor preprocessing (`tensor_preprocessing`), and PyTorch model execution (`pytorch_forward_pass`).
    - Exposed `X-Trace-ID` in HTTP response headers and verified with unit tests in `tests/test_telemetry.py`.
- **Deferred (To Be Done Later)**:
  - External notification webhook secrets provisioning (Slack / Discord / PagerDuty webhook URLs).

