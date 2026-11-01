# 🎓 Edge MLOps Engineering Masterclass: Memory-Constrained Vehicle Classifier
**A Complete, Stage-by-Stage Production Engineering Guide, Architecture Defense, and Professor Demo Manual**

---

## 📑 Table of Contents
1. [Executive Summary & Problem Formulation](#1-executive-summary--problem-formulation)
2. [End-to-End MLOps Lifecycle (Stage-by-Stage)](#2-end-to-end-mlops-lifecycle-stage-by-stage)
   - [Stage 1: Data Engineering, Versioning & Lineage (DVC + DagsHub)](#stage-1-data-engineering-versioning--lineage-dvc--dagshub)
   - [Stage 2: Model Architecture & Knowledge Distillation (Teacher -> Student)](#stage-2-model-architecture--knowledge-distillation-teacher---student)
   - [Stage 3: Experiment Tracking, Reproducibility & Model Registry (MLflow)](#stage-3-experiment-tracking-reproducibility--model-registry-mlflow)
   - [Stage 4: Automated Quality Gates & CI/CD Pipelines (GitHub Actions)](#stage-4-automated-quality-gates--cicd-pipelines-github-actions)
   - [Stage 5: Secure Containerization & Cloud Deployment (Docker + Azure AKS)](#stage-5-secure-containerization--cloud-deployment-docker--azure-aks)
   - [Stage 6: Real-Time Enterprise Observability (Prometheus + Grafana)](#stage-6-real-time-enterprise-observability-prometheus--grafana)
   - [Stage 7: In-Cluster Continuous Traffic Generator (Simulated Edge Fleet)](#stage-7-in-cluster-continuous-traffic-generator-simulated-edge-fleet)
   - [Stage 8: Interactive Distributed Tracing & Span Waterfall (OpenTelemetry)](#stage-8-interactive-distributed-tracing--span-waterfall-opentelemetry)
3. [Codebase Architecture & Functionality Map](#3-codebase-architecture--functionality-map)
4. [Architectural Trade-Offs & Key Technical Decisions](#4-architectural-trade-offs--key-technical-decisions)
5. [Future Scope & Current Resource Constraints](#5-future-scope--current-resource-constraints)
6. [Professor Cross-Examination Defense Guide (20 Critical Questions)](#6-professor-cross-examination-defense-guide-20-critical-questions)
7. [Step-by-Step Live Demo Script for the Professor (10-Minute Walkthrough)](#7-step-by-step-live-demo-script-for-the-professor-10-minute-walkthrough)

---

## 1. Executive Summary & Problem Formulation

### The Problem
Deploying computer vision deep neural networks onto edge devices (e.g., smart toll booths, roadside traffic cameras, edge micro-controllers like Raspberry Pi or NVIDIA Jetson Nano) requires balancing two conflicting objectives:
1. **High Classification Accuracy**: Accurately distinguishing vehicle types in real time under diverse lighting, orientations, and weather conditions.
2. **Extreme Resource Constraints**: Operating under strict hardware limits:
   - **Model Package Footprint**: Strictly **< 5,000,000 bytes (5.0 MB)** on disk and RAM.
   - **Inference SLA**: Sub-**25 ms** turnaround on commodity low-power single-core x86/ARM CPUs.
   - **Zero Cloud Leakage**: Autonomous in-pod execution with memory bounded to 512 MB.

### Target Classes (Fixed Label Vocabulary)
The model classifies input images into 5 mutually exclusive categories:
- `Car` (Sedans, SUVs, hatchbacks)
- `Bus` (Transit, coach, school buses)
- `Truck` (Freight, cargo, flatbeds)
- `Bike` (Motorcycles, scooters, bicycles)
- `None` (Background noise, empty roads, non-vehicle objects)

### Key Results
- **Model Checkpoint Size**: **2,665,366 bytes (~2.54 MB)** on disk (comfortably beneath the 5.0 MB budget).
- **p95 CPU Turnaround Latency**: **~18–24 ms** (comfortably beating the 25 ms SLA).
- **Architecture**: **ShuffleNet V2 (0.5x)** trained via **Knowledge Distillation** from an **EfficientNet-B2** teacher model.
- **Production Status**: Deployed as an auto-scaling cluster on Azure Kubernetes Service (AKS), actively monitored via Prometheus & Grafana, with live distributed tracing.

---

## 2. End-to-End MLOps Lifecycle (Stage-by-Stage)

```
+---------------------------------------------------------------------------------------------------------+
|                                        THE MLOPS LIFECYCLE                                              |
|                                                                                                         |
|  [Stage 1: DVC + DagsHub]      -->  [Stage 2: Knowledge Distillation] -->  [Stage 3: MLflow Registry]   |
|  Data Versioning & Splits            Teacher (31MB) -> Student (2.6MB)     Metrics, Artifacts & Champion|
|             |                                                                           |               |
|             v                                                                           v               |
|  [Stage 6: Observability]      <--  [Stage 5: Azure AKS Deployment]  <--  [Stage 4: CI/CD Quality Gate] |
|  Prometheus + Grafana + OTel        HPA, LoadBalancer, Non-Root OCI        Ruff, Pytest (27), Size Gate |
+---------------------------------------------------------------------------------------------------------+
```

---

### Stage 1: Data Engineering, Versioning & Lineage (DVC + DagsHub)

#### Objective
Raw imagery is large and binary; storing it directly inside Git bloats repository history. We need immutable data versioning, deterministic train/test splits, and remote synchronization.

#### Datasets Used
1. **MIO-TCD (Motorized and Non-Motorized Vehicle Dataset)**: Real-world surveillance traffic cameras from North American highways.
2. **ImageNet Subsets**: High-resolution negative samples and background noise for the `None` class.

#### How It Works
- **DVC (Data Version Control)** tracks pointers (`.dvc` files) containing SHA-256 and MD5 hashes of raw and processed folders:
  - `vehicle_dataset.dvc` (Train & Validation sets)
  - `test_dataset.dvc` (Curated test set)
  - `test_MIO_TCD_data.dvc` (Raw external benchmark)
- **DagsHub S3 Remote**: DVC uses an S3-compatible remote storage hosted on DagsHub:
  ```bash
  dvc remote add origin https://dagshub.com/DoshiVansh/Memory_constraint_Vehicle_classifier.s3.dvc
  dvc push -r origin
  ```
- **Data Pipeline Orchestration ([`dvc.yaml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/dvc.yaml))**:
  - `organize_data`: Scrapes raw folders and enforces balanced splits (70% train, 15% validation, 15% test).
  - `train_teacher`: Fits the heavy EfficientNet-B2 teacher network.
  - `distillation`: Fits the ShuffleNet-V2 student via KD loss.
  - `evaluate`: Generates confusion matrices, accuracy reports, and latency benchmarks.

---

### Stage 2: Model Architecture & Knowledge Distillation (Teacher -> Student)

#### Why Standard Training Fails on Constrained Hardware
Training a small network (e.g. SqueezeNet or MobileNet-0.25x) from scratch with hard one-hot labels leads to poor generalization. The network lacks parameter capacity to disentangle complex visual edge cases (e.g. differentiating a long pickup truck from a small bus).

#### The Knowledge Distillation Solution
We employ **Hinton's Knowledge Distillation**:
1. **Teacher Model ([`train_teacher.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/train_teacher.py))**:
   - Architecture: **EfficientNet-B2** (~31.3 MB, ~9.2M parameters).
   - High representational power, trained on ImageNet and fine-tuned on vehicle traffic.
2. **Student Model ([`train_distillation.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/train_distillation.py))**:
   - Architecture: **ShuffleNet V2 (0.5x channel scale)** (~2.66 MB, ~1.4M parameters).
   - Utilizes point-wise group convolutions, channel split, and channel shuffle operations to minimize memory access costs (MAC) on CPU caches.

#### The Loss Function
The student is trained using a composite objective:
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y_{\text{true}}, \sigma(z_s)) + (1 - \alpha) \cdot T^2 \cdot \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right), \sigma\left(\frac{z_t}{T}\right)\right)$$
- $z_s, z_t$: Logits produced by Student and Teacher.
- $T = 4.0$ (Temperature): Softens probability distributions, forcing the student to learn dark knowledge (e.g. how a Truck shares subtle features with a Bus).
- $\alpha = 0.3$ (Alpha weighting): 30% weight to ground-truth labels, 70% weight to teacher soft predictions.
- $T^2$ multiplier: Scales gradients to preserve magnitude during backpropagation.

#### Results Comparison
| Metric | Teacher (EfficientNet-B2) | Student (ShuffleNet-V2 0.5x) | Difference / Savings |
| :--- | :---: | :---: | :---: |
| **Model Size on Disk** | 31,290,163 bytes (~31.3 MB) | **2,665,366 bytes (~2.54 MB)** | **91.5% reduction** |
| **CPU Inference Latency** | ~85–110 ms | **~11–18 ms** | **5x faster** |
| **Test Accuracy** | ~92.4% | **~89.8%** | Only ~2.6% trade-off |
| **5 MB Hard Budget** | ❌ Fails (6x over limit) | ✅ **Passes (53% margin)** | Compliant |

---

### Stage 3: Experiment Tracking, Reproducibility & Model Registry (MLflow)

#### How It Works
- **Tracking Server**: All experiments log to MLflow (`mlflow.db` locally and synchronized to DagsHub remote MLflow server).
- **Hyperparameter Logging**:
  - `batch_size: 64`, `learning_rate: 0.001`, `optimizer: AdamW`, `temperature: 4.0`, `alpha: 0.3`.
- **Metrics Tracked**:
  - `train_loss`, `val_loss`, `distillation_loss`, `student_val_acc`, `teacher_val_acc`, `inference_latency_ms`.
- **Model Lineage Manifest ([`student_model.pth.json`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/student_model.pth.json))**:
  Every packaged checkpoint is paired with an immutable JSON file recording:
  ```json
  {
    "model_name": "student_model",
    "model_version": "1",
    "model_alias": "champion",
    "architecture": "shufflenet_v2_x0_5",
    "artifact_sha256": "c80eaf7a45e87e604b2e41af992094425834e46d747315a6fb66a65f7c18e1d4",
    "size_bytes": 2665366,
    "max_allowed_bytes": 5000000,
    "num_classes": 5,
    "class_names": ["Bus", "Car", "None", "Truck", "Bike"]
  }
  ```
- **Champion Tagging**:
  Only the model checkpoint tagged with `@champion` in the registry and satisfying the `< 5,000,000 bytes` safety constraint is baked into release containers.

---

### Stage 4: Automated Quality Gates & CI/CD Pipelines (GitHub Actions)

#### Pipeline Flowchart
```
Git Push Tag (v1.0.3)
   ├── Job 1: Quality Gate (ci.yml)
   │     ├── Static Linting: ruff check
   │     ├── Type Safety: mypy validation
   │     ├── Model Budget Gate: check size < 5,000,000 bytes
   │     └── Unit & Contract Tests: pytest (27 tests passing)
   │
   ├── Job 2: Build & Publish OCI Container (release-image.yml)
   │     ├── Multi-stage Docker build
   │     ├── GHCR login & image push: ghcr.io/...:v1.0.3
   │     └── Write release lineage attestation
   │
   └── Job 3: Automated Staging Deployment (deploy.yml)
         ├── Decode AKS Kubeconfig via setup_kubeconfig.py
         ├── Render Kubernetes manifests with image SHA
         ├── kubectl apply: deployment, service, hpa, configmap
         └── Rolling update zero-downtime health check
```

#### Key Workflows
1. [`.github/workflows/ci.yml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/.github/workflows/ci.yml):
   - Runs automated unit tests, validating:
     - `/health/live` and `/health/ready` return 200.
     - `/predict` returns class, confidence, latency, and valid W3C `X-Trace-ID`.
     - Invalid MIME types or empty files return HTTP 400 with structured JSON error details.
     - Model size strictly matches the SHA-256 integrity digest.
2. [`.github/workflows/release-image.yml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/.github/workflows/release-image.yml):
   - Runs upon tagging a release (`git tag v1.0.3 && git push origin v1.0.3`).
   - Packages dependencies and embeds champion weights into a minimal OCI image.
   - Pushes to GitHub Container Registry (GHCR): `ghcr.io/doshivansh/memory_constraint_vehicle_classifier:v1.0.3`.
3. [`.github/workflows/deploy.yml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/.github/workflows/deploy.yml):
   - Connects to Azure AKS cluster using sanitized base64 credentials ([`scripts/setup_kubeconfig.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/scripts/setup_kubeconfig.py)).
   - Executes rolling update of Kubernetes deployments without downtime.

---

### Stage 5: Secure Containerization & Cloud Deployment (Docker + Azure AKS)

#### Container Hardening ([`Dockerfile`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/Dockerfile))
- **Two-Stage Multi-Stage Build**:
  - `builder`: Downloads PyTorch CPU wheels (`torch`, `torchvision` CPU variants, skipping CUDA to save 2.5 GB of image overhead).
  - `runtime`: Minimal `python:3.10-slim` image (~210 MB total container size).
- **Security Posture**:
  - Runs as unprivileged user `appuser` (UID `10001`).
  - `readOnlyRootFilesystem: true` enforced in Kubernetes.
  - Drops all Linux kernel capabilities (`capabilities: drop: [ALL]`).
  - Only `/tmp` mounted as in-memory emptyDir.

#### Kubernetes Production Cluster Layout
- **Cloud Provider**: Azure Kubernetes Service (AKS).
- **Namespace**: `vehicle-classifier-staging`.
- **Workloads**:
  1. `vehicle-classifier-api`: 2 pod replicas behind an Azure LoadBalancer (`20.200.237.154:8000`).
  2. `vehicle-classifier-prometheus`: Prometheus server (`10.0.64.193:9090`) scraping API every 5 seconds.
  3. `vehicle-classifier-grafana`: Public Grafana dashboard (`20.249.161.211:3000`).
  4. `vehicle-classifier-traffic-generator`: In-cluster synthetic load generator simulating edge requests.
- **Autoscaling ([`k8s/hpa.yaml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/k8s/hpa.yaml))**:
  Horizontal Pod Autoscaler monitors CPU utilization:
  - Min Replicas: 2 | Max Replicas: 10 | Target CPU: 70%.

---

### Stage 6: Real-Time Enterprise Observability (Prometheus + Grafana)

#### Metrics Architecture
The API instruments custom Prometheus metrics using a fixed label vocabulary (preventing label cardinality explosion) in [`vehicle_classifier_core/api.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/api.py):
- `vehicle_classifier_requests_total{endpoint, method, status}`: Request counter.
- `vehicle_classifier_predictions_total{model_version, predicted_class}`: Prediction counter per class.
- `vehicle_classifier_request_duration_seconds_bucket`: Latency histogram with buckets tuned from 5ms to 100ms.
- `vehicle_classifier_inference_duration_seconds_bucket`: Pure PyTorch forward-pass latency histogram.
- `vehicle_classifier_errors_total{endpoint, error_type}`: Error counter.
- `vehicle_classifier_in_progress_requests`: Active concurrent request gauge.

#### Grafana Enterprise Dashboard
Accessed publicly at: **[http://20.249.161.211:3000/d/vehicle-classifier-overview](http://20.249.161.211:3000/d/vehicle-classifier-overview)** *(admin / admin)*

The dashboard contains 16 auto-refreshing panels arranged in 5 analytical tiers:
1. **Tier 1 (Executive KPIs)**:
   - *Total Inferences Served*: Live running counter across pods.
   - *Prediction Throughput*: Instant req/sec rate.
   - *p95 Latency SLA*: 95th percentile turnaround time (Target < 25 ms, current ~18–24 ms).
   - *HTTP Success Rate*: Percentage of 200 OK responses vs 4xx/5xx errors (Current ~96%).
   - *Active API Replicas*: Count of healthy pods scraped by Prometheus (`up == 1`).
   - *Champion Model Info*: Loaded model version and architecture verification.
2. **Tier 2 (Traffic & Prediction Breakdown)**:
   - *Throughput Timeseries*: Real-time req/sec traffic curve.
   - *Vehicle Class Predictions Donut*: Percentage breakdown across `Car`, `Bus`, `Truck`, `Bike`, `None`.
   - *Class Predictions Over Time*: Stacked timeseries showing traffic composition trends.
3. **Tier 3 (Latency & SLA Performance)**:
   - *End-to-End Latency Percentiles*: Simultaneous tracking of p50 (Median), p90, p95 (SLA limit), and p99 (Tail).
   - *Pure Model Inference vs API Network Overhead*: Contrasts PyTorch forward pass (~11ms) against image decode + color convert + HTTP transmission (~3ms).
4. **Tier 4 (Infrastructure & Container Health)**:
   - *API Container CPU Usage*: CPU core consumption rate.
   - *API Resident Memory (RAM RSS)*: Bounded memory tracking (~330 MB vs 512 MB request).
   - *Active In-Flight Requests*: Real-time concurrency gauge.
   - *HTTP Status Distribution*: Stacked graph of 200s vs 400s vs 500s.
5. **Tier 5 (OpenTelemetry Tracing Guide)**:
   - Embedded interactive documentation and direct launch links to the waterfall UI.

---

### Stage 7: In-Cluster Continuous Traffic Generator (Simulated Edge Fleet)

#### Why It's Needed
In an edge camera deployment, vehicles pass by continuously. Without continuous traffic, Prometheus metrics would flatline and the Grafana dashboard would show "No data" during a demonstration.

#### Implementation ([`scripts/traffic_generator.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/scripts/traffic_generator.py))
- Runs inside the cluster as pod `vehicle-classifier-traffic-generator`.
- Cycles through high-resolution base64 embedded templates and test dataset files (`Car`, `Bus`, `Truck`, `Bike`, `None`).
- Adds realistic augmentations on the fly:
  - Random contrast adjustments (0.7x – 1.3x).
  - Random sharpness adjustments (0.5x – 1.5x).
  - Gaussian blur and sensor noise.
- Injects periodic edge-case probes:
  - 4% of traffic sends non-image payloads to verify HTTP 400 error handling and alert rules.
- Emits structured logs with latency and W3C trace IDs:
  ```bash
  kubectl logs -l app.kubernetes.io/name=vehicle-classifier-traffic-generator -n vehicle-classifier-staging --tail=10
  ```

---

### Stage 8: Interactive Distributed Tracing & Span Waterfall (OpenTelemetry)

#### Why APM Tracing Matters in MLOps
When an API call takes 25ms, a machine learning engineer needs to know:
- Did the delay come from network upload buffering?
- Did image decompression take too long because someone sent an uncompressed BMP?
- Did PyTorch forward pass slow down due to CPU throttling?

#### Implementation Architecture
1. **Tracing Engine ([`vehicle_classifier_core/telemetry.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/telemetry.py))**:
   - Implements OpenTelemetry Tracer adhering to the **W3C Trace Context** specification.
   - Every response injects an `X-Trace-ID` header (32-character hexadecimal string).
   - Implements an `InMemoryTraceStore` (inheriting from `SpanExporter`) that buffers the last 500 execution trees in RAM without requiring an expensive external database cluster.
2. **Interactive Waterfall UI ([`vehicle_classifier_core/trace_dashboard_html.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/trace_dashboard_html.py))**:
   - Accessible at: **[http://20.200.237.154:8000/traces](http://20.200.237.154:8000/traces)**.
   - **Left Sidebar**: Real-time thread and request feed with class badges (`Car`, `Bike`, etc.), durations, and timestamps. Auto-polls every 3 seconds.
   - **Right Main Pane**: Visual Gantt waterfall showing the exact hierarchical breakdown:
     ```
     predict_request (HTTP POST /predict - Root Span) [~14.3 ms]
       ├── [01] read_upload_payload          (~0.03 ms) [Reads multipart stream, validates size]
       ├── [02] validate_and_decode_image    (~0.15 ms) [PIL decompression & format verification]
       ├── [03] convert_image_rgb            (~0.25 ms) [Transforms grayscale/RGBA to 3-channel RGB]
       └── [04] model_inference              (~13.3 ms) [Inference orchestrator]
             ├── tensor_preprocessing        (~1.37 ms) [Normalizes pixels & resizes to 224x224]
             └── pytorch_forward_pass        (~11.4 ms) [ShuffleNet V2 computation graph]
     ```
   - **Span Inspector**: Clicking any bar displays microsecond durations, parent span ID, and span attributes (`image.format`, `image.pixels`, `prediction.class_name`, `prediction.confidence`, `tensor.shape`, `model.version`, `model.sha256`).

---

## 3. Codebase Architecture & Functionality Map

Use this table during cross-examination to pinpoint exactly where any logic lives:

| Functionality | File Path | Key Classes / Functions / Section |
| :--- | :--- | :--- |
| **Model Architecture** | [`vehicle_classifier_core/models.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/models.py) | `build_shufflenet_v2_x0_5()`, `build_efficientnet_b2()` |
| **Inference Engine** | [`vehicle_classifier_core/predictor.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/predictor.py) | `VehiclePredictor.predict()`, `preprocess_image()`, PyTorch inference |
| **REST API & Metrics** | [`vehicle_classifier_core/api.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/api.py) | `FastAPI`, `PrometheusMetricsMiddleware`, `/predict`, `/metrics`, `/traces` |
| **Distributed Tracing** | [`vehicle_classifier_core/telemetry.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/telemetry.py) | `InMemoryTraceStore`, `init_telemetry()`, `get_trace_store()` |
| **Waterfall UI HTML** | [`vehicle_classifier_core/trace_dashboard_html.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/trace_dashboard_html.py) | `render_trace_dashboard_html()` (Single-page reactive UI) |
| **Knowledge Distillation** | [`train_distillation.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/train_distillation.py) | `DistillationLoss`, training loop with temperature $T$ |
| **Teacher Training** | [`train_teacher.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/train_teacher.py) | Cross-entropy training on EfficientNet-B2 |
| **Model Size Gate** | [`scripts/create_release_metadata.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/scripts/create_release_metadata.py) | `MAX_ALLOWED_BYTES = 5000000`, SHA-256 verification |
| **DVC Pipeline** | [`dvc.yaml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/dvc.yaml) | Pipeline stages, inputs/outputs, parameter tracking |
| **Continuous Traffic** | [`scripts/traffic_generator.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/scripts/traffic_generator.py) | Augmented requests across all 5 classes |
| **Grafana Generator** | [`scripts/generate_grafana_dashboard.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/scripts/generate_grafana_dashboard.py) | 16-panel dashboard JSON generator with instant PromQL |
| **Kubernetes API** | [`k8s/deployment.yaml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/k8s/deployment.yaml) | Replicas, resource limits, security context, readiness probes |
| **Autoscaling (HPA)** | [`k8s/hpa.yaml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/k8s/hpa.yaml) | CPU-based horizontal autoscaling (min 2, max 10) |
| **Kubeconfig Auth** | [`scripts/setup_kubeconfig.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/scripts/setup_kubeconfig.py) | Base64 stripping, auto-padding, schema verification |
| **Unit & Contract Tests** | [`tests/test_api.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/tests/test_api.py) | 27 passing tests (contracts, endpoints, errors, tracing) |

---

## 4. Architectural Trade-Offs & Key Technical Decisions

Be prepared to explain these design trade-offs during your defense:

### 1. ShuffleNet V2 vs. MobileNetV3 vs. SqueezeNet
- **Why ShuffleNet V2 Wins**: SqueezeNet has small weights (~5 MB), but its fire modules require heavy DRAM access, causing CPU cache misses. MobileNetV3 uses depthwise separable convolutions with squeeze-and-excitation blocks which introduce non-negligible latency on low-end single-core CPUs.
- **The ShuffleNet Advantage**: ShuffleNet V2 was designed according to practical guidelines for efficient network design (Ma et al., ECCV 2018): equal channel width minimizes memory access cost (MAC), and channel shuffle operations run in $O(1)$ cache time.
- **The Trade-off**: Slightly lower theoretical parameter count efficiency than MobileNetV3 in exchange for significantly faster actual wall-clock execution on commodity CPUs.

### 2. PyTorch State Dict vs. TorchScript / ONNX
- **Decision**: Serving via PyTorch state-dict with frozen evaluation mode (`torch.no_grad()`).
- **The Trade-off**: While ONNX Runtime or TensorRT could yield another 2–3 ms latency reduction, they require additional C++ shared libraries (`onnxruntime.so`, CUDA drivers) that would balloon the Docker image by 300+ MB and violate pure portable CPU constraints. Pure PyTorch CPU is reproducible across any Linux container without driver dependencies.

### 3. In-Memory Trace Store vs. Jaeger / Tempo Cluster
- **Decision**: Implementing `InMemoryTraceStore` directly in the API container.
- **The Trade-off**: An enterprise APM database (like Grafana Tempo or Jaeger with Cassandra) would require 3 additional pods, persistent volumes, and ~1.5 GB of RAM. Our in-memory ring buffer (500 traces) delivers sub-millisecond trace inspection and a zero-infrastructure footprint.

### 4. Prometheus `/metrics/` ASGI Mount vs. Custom Route
- **Decision**: Using `prometheus_client.make_asgi_app()` mounted at `/metrics`.
- **The Trade-off**: FastAPI's route matching issues a 307 redirect from `/metrics` to `/metrics/`. We configured Prometheus scrape configs to scrape `/metrics/` with trailing slash, avoiding redirect latency.

---

## 5. Future Scope & Current Resource Constraints

When asked *"What is missing?"* or *"What would you build next?"*, deliver this structured answer:

### A. Current Stack Limitations (Why it wasn't done now)
1. **Compute Resources**: The staging cluster runs on standard Azure AKS B-series / D-series VMs without dedicated edge accelerators (NVIDIA Jetson, Google Coral Edge TPU, Intel Neural Compute Stick).
2. **Storage Budget**: We deliberately avoided spinning up elastic block stores (PV/PVCs) for Jaeger/Tempo to keep cloud hosting costs minimal for academic staging.
3. **Network Architecture**: The API currently terminates TLS via Azure LoadBalancer rather than an enterprise Ingress Controller (e.g. Traefik or NGINX Ingress) to minimize reverse-proxy hops.

### B. Phase 2 Roadmap (Future Scope)
1. **Dynamic Quantization (INT8)**: Post-training quantization using PyTorch FX Graph Mode Quantization to shrink weights from 2.66 MB (FP32) to ~700 KB (INT8) and unlock 2x SIMD CPU speedups.
2. **Continuous Drift Detection (Evidently AI)**: Integrating an automated background job monitoring Kolmogorov-Smirnov (KS) test statistics on incoming pixel distributions to detect lighting and camera lens drift.
3. **Automated Retraining Loop (GitOps)**: Triggering a GitHub Actions / DVC repro workflow when class distribution shifts by > 20% over a 7-day rolling window.
4. **Service Mesh & Canary Releases (Istio)**: Splitting traffic (90% to Champion v1.0.3, 10% to Candidate v1.0.4) using header routing and automated rollback upon SLA breach.

---

## 6. Professor Cross-Examination Defense Guide (20 Critical Questions)

### Q1: Why did you use Knowledge Distillation instead of just training the small model directly?
> **Answer**: Direct training of ShuffleNet V2 with one-hot cross-entropy produces ~83% accuracy because the small model lacks the capacity to discover nuanced decision boundaries. By using Knowledge Distillation with Temperature $T=4.0$, the student learns the *dark knowledge* (soft class probabilities) of the 31MB teacher. This boosted student accuracy to **89.8%** while preserving a **2.66 MB** footprint.

### Q2: What exactly ensures your model never exceeds the 5 MB constraint in production?
> **Answer**: We have an immutable **Quality Gate** in CI/CD. The script [`scripts/create_release_metadata.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/scripts/create_release_metadata.py) checks `os.path.getsize(model_path) < 5,000,000`. If someone trains a larger model, the CI pipeline fails immediately, blocking container image publishing and deployment.

### Q3: How do you prevent data leakage between your train, validation, and test splits?
> **Answer**: In [`dataset_organizer.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/dataset_organizer.py), splitting is deterministic using fixed random seeds (`seed=42`). Image augmentations (jitter, blur, rotation) are applied **strictly to the training split** at runtime via PyTorch transforms; validation and test splits only undergo deterministic resizing and standard ImageNet normalization.

### Q4: How is data versioning handled if you cannot store gigabytes of images in Git?
> **Answer**: We use **DVC** paired with an S3-compatible remote on DagsHub. Git only tracks lightweight pointer files (`vehicle_dataset.dvc`) containing MD5 checksums. The actual images are downloaded deterministically using `dvc pull`.

### Q5: What is the purpose of the `None` class? Why not just return low confidence?
> **Answer**: In production traffic cameras, empty roads, pedestrians, shadows, and debris frequently trigger the camera. Without an explicit `None` class trained on negative ImageNet samples, the softmax layer is forced to assign 100% of the probability mass across vehicles, causing severe false positives.

### Q6: Why did you choose ShuffleNet V2 over MobileNet?
> **Answer**: While MobileNet has high theoretical FLOP efficiency, its depthwise convolutions have low arithmetic intensity (FLOPs per byte of memory access). ShuffleNet V2 enforces equal channel widths and channel shuffles, minimizing Memory Access Cost (MAC), which translates directly to lower wall-clock latency on CPU caches.

### Q7: What is the role of temperature ($T$) in your distillation loss function?
> **Answer**: Standard softmax produces sharp probabilities (e.g. $[0.98, 0.01, 0.01]$). Applying $T=4.0$ softens the distribution (e.g. $[0.60, 0.25, 0.15]$), revealing the relative similarities between classes (e.g. that a Bus looks more like a Truck than a Bike). The student learns these structural correlations.

### Q8: Why did your Grafana Stat panels initially show "No data" and "Load failed"?
> **Answer**: Two distinct issues occurred: First, a query contained `or 0`, which caused Prometheus to return HTTP 400 because PromQL logical `or` requires two vector operands. Second, Grafana `stat` panels were executing range queries without `"instant": true`, leading to timestamp mismatches on counters. Setting `"instant": true` and fixing vector fallback arithmetic solved both.

### Q9: Why was the class prediction donut chart showing only "Bike 66.4 4%" repeatedly?
> **Answer**: The pie chart was set to `"values": true` ("All Values" mode) rather than `"values": false` ("Calculate" mode). In All Values mode, Grafana rendered an individual slice for every timestamp in the 15-minute range query (25 timestamps $\times$ 4% = 100%). Changing to `"values": false` and `"instant": true` groups by the `predicted_class` label correctly.

### Q10: How does your deployment handle container crashes or memory leaks?
> **Answer**: Kubernetes enforces three probes in [`k8s/deployment.yaml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/k8s/deployment.yaml):
> 1. `startupProbe`: Allows up to 90s for model weight loading.
> 2. `livenessProbe`: Restarts the pod if `/health/live` fails 3 consecutive times.
> 3. `readinessProbe`: Removes unready pods from the LoadBalancer rotation.
> Additionally, memory is capped at 2 GiB with a 512 MiB request to prevent node OOM events.

### Q11: How do you achieve zero-downtime deployments when rolling out a new model version?
> **Answer**: The Kubernetes deployment uses `RollingUpdate` with `maxUnavailable: 0` and `maxSurge: 1`. Kubernetes spins up a new pod with the updated model, verifies its readiness probe succeeds, and only then terminates the old replica.

### Q12: Why did you mount `/metrics/` with a trailing slash?
> **Answer**: In Starlette/FastAPI, mounting an ASGI application at `/metrics` redirects requests without a trailing slash to `/metrics/` with HTTP 307. Scrapers that do not follow redirects fail. Configuring the Prometheus scrape config with `metrics_path: /metrics/` eliminates the redirect hop.

### Q13: What security measures are implemented inside the container?
> **Answer**: The container runs as non-root (`appuser`, UID 10001), drops all Linux capabilities (`drop: [ALL]`), disallows privilege escalation, and mounts the root filesystem as read-only (`readOnlyRootFilesystem: true`). Only `/tmp` is writeable via in-memory storage.

### Q14: How do you correlate Prometheus metrics with specific distributed traces?
> **Answer**: The API injects a W3C-compliant `X-Trace-ID` into every HTTP response header. When an alert or latency spike occurs in Prometheus/Grafana, an engineer copies the Trace ID or timestamp range and inspects the exact execution waterfall in the tracing dashboard at `/traces`.

### Q15: What happens if a user uploads a corrupted 100MB video file instead of an image?
> **Answer**: The API has strict defensive bounds in [`vehicle_classifier_core/api.py`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/api.py):
> 1. Payload size is validated in chunks and rejected immediately with HTTP 413 if it exceeds `MAX_UPLOAD_BYTES` (8 MB).
> 2. The MIME type is checked, and image headers are decompressed using PIL within a protected span. If corrupted, it returns HTTP 400 Bad Request without crashing the worker process.

### Q16: How does Horizontal Pod Autoscaling (HPA) decide when to scale out?
> **Answer**: The HPA ([`k8s/hpa.yaml`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/k8s/hpa.yaml)) continuously polls the Kubernetes Metrics Server. When average pod CPU utilization exceeds **70%** of the 250m request threshold, HPA scales the deployment from 2 up to a maximum of 10 replicas.

### Q17: Why did you choose OpenTelemetry over vendor-specific APMs like Datadog or New Relic?
> **Answer**: OpenTelemetry is the vendor-neutral CNCF open standard. It prevents vendor lock-in, adheres to the W3C Trace Context specification, and can export to Jaeger, Tempo, Zipkin, or Datadog without rewriting application instrumentation.

### Q18: What is the purpose of the traffic generator pod in the cluster?
> **Answer**: In production, camera feeds are asynchronous. In our demonstration environment, the traffic generator simulates edge camera activity by sending continuous, augmented requests with realistic delays, ensuring metrics, histograms, and traces remain live and observable.

### Q19: How do you verify the integrity of the model checkpoint running inside the container?
> **Answer**: We record the SHA-256 digest in [`student_model.pth.json`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/student_model.pth.json). Upon startup, [`VehiclePredictor`](file:///Users/naitikbhavsar/IIIT%20Bangalore/MLops%20PE/Memory_constraint_Vehicle_classifier-vansh_mlops/vehicle_classifier_core/predictor.py) reads the model bytes, computes the SHA-256 hash, and compares it against metadata. If a byte is corrupted or altered, the startup probe fails.

### Q20: How would you extend this architecture to support video streams (e.g. RTSP)?
> **Answer**: We would place a lightweight frame-sampling sidecar container (using OpenCV/GStreamer) that ingests the RTSP video stream, drops redundant frames (e.g. 1 frame every 500ms), and dispatches crops over local Unix domain sockets to the `/predict` endpoint.

---

## 7. Step-by-Step Live Demo Script for the Professor (10-Minute Walkthrough)

Follow this proven, highly structured demo script:

### Act 1: The Pitch & Architecture Overview (Time: 0:00 – 1:30)
- **Say**: *"Professor, in this project we tackled the challenge of deploying deep vision models to memory-constrained edge hardware. Our strict constraints were: model footprint under 5 Megabytes, and CPU inference under 25 milliseconds."*
- **Action**: Open GitHub repository. Highlight the two checkpoints:
  - Teacher: `EfficientNet_B2_teacher_model.pth` (**~31.3 MB**)
  - Student: `student_model.pth` (**~2.54 MB** - 91.5% smaller!)
- **Key Point**: Explain Knowledge Distillation: Student trained with Temperature $T=4.0$ to distill the teacher's soft probabilities into a lightweight ShuffleNet V2 backbone.

### Act 2: Interactive API & Live Prediction (Time: 1:30 – 3:30)
- **Say**: *"Let's test the live production API running on our Azure Kubernetes Service cluster."*
- **Action**: Open Swagger Docs: **[http://20.200.237.154:8000/docs](http://20.200.237.154:8000/docs)**.
- **Action**: Expand `POST /predict`. Click **Try it out**.
- **Action**: Upload an image from `test_dataset/Car/` or `test_dataset/Bike/`.
- **Action**: Click **Execute**. Show the JSON response:
  - `class_name`: "Car"
  - `confidence`: 0.94
  - `latency_ms`: 14.2 ms (Beating the 25ms SLA!)
  - `X-Trace-ID` in headers.

### Act 3: Interactive Distributed Tracing & Waterfall UI (Time: 3:30 – 5:30)
- **Say**: *"Now let's see what happened inside the container during that request using our OpenTelemetry APM tracing dashboard."*
- **Action**: Open: **[http://20.200.237.154:8000/traces](http://20.200.237.154:8000/traces)**.
- **Action**: Show the live thread stream on the left auto-refreshing with requests (`Car`, `Bike`, `Truck`, `None`).
- **Action**: Click the top request. Point to the Gantt waterfall chart on the right:
  - Root: `predict_request` (~14ms)
  - `read_upload_payload` (~0.03ms)
  - `validate_and_decode_image` (~0.15ms)
  - `convert_image_rgb` (~0.25ms)
  - `model_inference` -> `pytorch_forward_pass` (~11.4ms)
- **Action**: Click on `pytorch_forward_pass` to open the Span Inspector Card, showing exact model version, SHA-256 digest, and confidence attributes.
- **Key Point**: *"This gives complete engineering transparency into overhead vs pure compute."*

### Act 4: Real-Time Grafana Production Monitoring (Time: 5:30 – 8:00)
- **Say**: *"Let's look at enterprise cluster health on our live Grafana monitoring portal."*
- **Action**: Open: **[http://20.249.161.211:3000/d/vehicle-classifier-overview](http://20.249.161.211:3000/d/vehicle-classifier-overview)**.
- **Walk through the panels**:
  1. **Top Stat Row**: Point to Total Inferences (~700+), Throughput (~0.5 req/s), p95 Latency SLA (~18–24 ms), and HTTP Success Rate (~96%).
  2. **Class Distribution Donut**: Point to the live breakdown across `Car`, `Bike`, `Truck`, `None`.
  3. **End-to-End Latency Timeseries**: Point to p50, p90, p95, p99 lines proving consistent sub-25ms response times.
  4. **Model vs Overhead Graph**: Shows that network overhead is less than 3ms.
  5. **Infrastructure Health**: Shows API container CPU cores and RAM RSS (~330 MB) well within the 512 MB request limit.

### Act 5: Cloud Orchestration, GitOps & CI/CD Gates (Time: 8:00 – 10:00)
- **Say**: *"Finally, let's inspect the automated Kubernetes infrastructure and CI/CD quality gates."*
- **Action**: Open Terminal and run:
  ```bash
  kubectl get pods,svc,hpa -n vehicle-classifier-staging
  ```
  Show that 2 API replicas, Prometheus, Grafana, and the Traffic Generator are healthy and load-balanced.
- **Action**: Open GitHub Actions: Show the passing workflow runs for tag **`v1.0.3`**:
  - `Publish inference image`: Built multi-stage Docker image and published to GHCR.
  - `Continuous Deployment & Staging Verification`: Executed zero-downtime rolling update.
- **Concluding Sentence**: *"In summary, we have built a complete, production-grade MLOps platform: from data versioning and knowledge distillation, to hardened container deployment, autoscaling, and real-time distributed observability."*
