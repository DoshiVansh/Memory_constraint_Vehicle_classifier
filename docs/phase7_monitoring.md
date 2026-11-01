# Phase 7 monitoring and local performance run

## Monitoring design

The inference API exposes Prometheus metrics at `/metrics`:

| Metric | Meaning |
|---|---|
| `vehicle_classifier_requests_total{method,endpoint,status}` | Count of non-health HTTP requests, using HTTP method, fixed endpoint (`/predict` or `unmatched`), and status code. |
| `vehicle_classifier_errors_total{endpoint,error_type}` | 4xx and 5xx counts grouped by endpoint and error class. Raw exception text is never a label. |
| `vehicle_classifier_request_duration_seconds{endpoint}` | End-to-end HTTP request time, including upload parsing and image decoding. |
| `vehicle_classifier_in_progress_requests{endpoint}` | Requests currently in progress. |
| `vehicle_classifier_inference_duration_seconds{model_version}` | Time spent inside the classifier call. |
| `vehicle_classifier_predictions_total{predicted_class,model_version}` | Successful predictions for one of the five fixed classes, grouped by model version. |
| `vehicle_classifier_model_info{model_version,architecture}` | Identity of the model loaded in this API process. |
| `process_cpu_seconds_total`, `process_resident_memory_bytes` | CPU time and resident memory for the API process. These are process values, not full pod/cgroup metrics. |

The middleware deliberately excludes `/metrics` and readiness/liveness probes
from request totals. It uses only route templates and a fixed set of labels;
image names, input data, arbitrary paths, client identifiers, and exception
messages never become labels. Class counts are an unlabeled-input drift signal,
not evidence of accuracy or concept drift. Measuring quality still requires
later labeled feedback.

Prometheus scrapes the API every five seconds, retains up to seven days in an
ephemeral volume, evaluates two warning rules, and stores data only in this
local demo cluster. The Grafana dashboard is provisioned from files and shows
traffic, 5xx rate, p50/p95 HTTP latency, p95 model-call latency, class
distribution, API process CPU, API process memory, and in-flight requests.
Both services are ClusterIP-only. Grafana is configured for anonymous viewer
access for this local port-forward demo; do not expose it on a public network.
Prometheus alert rules are visible in its UI, but no Alertmanager/contact point
is configured to send notifications.

The 5% 5xx and one-minute target-down rules are temporary demo warnings. They
are not service-level objectives. Choose SLO/alert thresholds only after
measuring on the hardware and network that will actually serve users.

## Start the local monitoring stack

The API image must include the metrics code. On the provided kind cluster:

```bash
docker build -t vehicle-classifier-api:phase7 .
kind load docker-image vehicle-classifier-api:phase7 --name vehicle-classifier-phase6
```

For this local-only cluster demo, re-render the deployment with the local image
and `imagePullPolicy: Never` (the Jenkins release template otherwise expects an
immutable GHCR digest):

```bash
export IMAGE_REF=vehicle-classifier-api:phase7
export MODEL_VERSION=2
export RELEASE_GIT_SHA="$(git rev-parse HEAD)"
envsubst '${IMAGE_REF} ${MODEL_VERSION} ${RELEASE_GIT_SHA}' < k8s/deployment.yaml.tpl \
  | sed 's/imagePullPolicy: IfNotPresent/imagePullPolicy: Never/' \
  | kubectl --context kind-vehicle-classifier-phase6 apply -f -
kubectl --context kind-vehicle-classifier-phase6 rollout status \
  deployment/vehicle-classifier-api -n vehicle-classifier-staging --timeout=5m
```

Apply Prometheus and Grafana; the pinned monitoring images must be pullable by
the kind node:

```bash
kubectl --context kind-vehicle-classifier-phase6 apply -f k8s/monitoring/prometheus.yaml
kubectl --context kind-vehicle-classifier-phase6 apply -f k8s/monitoring/grafana.yaml
kubectl --context kind-vehicle-classifier-phase6 rollout status \
  deployment/vehicle-classifier-prometheus -n vehicle-classifier-staging --timeout=5m
kubectl --context kind-vehicle-classifier-phase6 rollout status \
  deployment/vehicle-classifier-grafana -n vehicle-classifier-staging --timeout=5m
```

Forward the internal Services in separate terminals:

```bash
kubectl --context kind-vehicle-classifier-phase6 port-forward \
  -n vehicle-classifier-staging service/grafana 3000:3000
kubectl --context kind-vehicle-classifier-phase6 port-forward \
  -n vehicle-classifier-staging service/prometheus 9090:9090
kubectl --context kind-vehicle-classifier-phase6 port-forward \
  -n vehicle-classifier-staging service/vehicle-classifier-api 8000:8000
```

Open Grafana at <http://127.0.0.1:3000/d/vehicle-classifier-overview> (viewer,
no login for this local-only demo). Prometheus is at <http://127.0.0.1:9090>;
check **Status → Targets** and **Alerts**. The API metrics exposition is at
<http://127.0.0.1:8000/metrics>.

## Repeat the controlled test

From the repository root with API port-forwarding active:

```bash
python3 scripts/load_test.py \
  --base-url http://127.0.0.1:8000 \
  --concurrency 1 2 4 \
  --requests-per-level 100 \
  --service-image-ref vehicle-classifier-api:phase7 \
  --cluster-context kind-vehicle-classifier-phase6 \
  --output reports/phase7_local_load.json
```

The script warms up with three requests, creates the same deterministic PNG for
all requests, and records p50/p95/max request latency, throughput, errors,
sampled API-process CPU/memory, client OS/Python/CPU/RAM, model hash, Git SHA,
and whether the working tree is dirty. Latency includes the local port-forward
and the service's upload/decode path. The input is a synthetic image and the
client runs on the same host as kind; use the result as a reproducible demo,
not as a remote-user or production capacity estimate.

## Measured local result

The retained run used concurrency 1, 2, and 4 with 100 requests at each level
on the local kind cluster. It returned 300/300 valid predictions with zero
errors. The measurements were:

| Concurrent requests | p50 latency | p95 latency | Throughput | API process CPU average | API process RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 134 ms | 155 ms | 7.88 req/s | 1.99 cores | 301.7 MiB |
| 2 | 202 ms | 280 ms | 9.53 req/s | 2.02 cores | 301.7 MiB |
| 4 | 401 ms | 409 ms | 9.98 req/s | 2.14 sampled cores | 301.7 MiB |

This suggests CPU saturation near the Deployment's two-core limit: more
concurrency increased latency without materially increasing throughput. This
is a synthetic, same-host sample (100 requests per concurrency), so it is not
an SLO or a claim about deployment capacity. CPU is estimated from samples
every 0.5 seconds and can vary around the configured two-core limit due to
sampling boundaries. The complete run, including class counts and raw latency
statistics, is stored in `reports/phase7_local_load.json`. The repeated
synthetic image is expected to produce a concentrated predicted-class
distribution and is not a representative population/drift reference.
The per-level prediction counts were `None: 100`; that says what this synthetic
image was classified as, not that the model's production class distribution is
known. A separate malformed upload was rejected with HTTP 422 and subsequently
appeared in Prometheus as one bounded `http_4xx` error. It was a metric check,
not part of the 300 successful load requests.

## Cleanup

The API, monitoring databases, and Grafana state are local to the disposable
kind cluster. Keep the cluster for the next phase, or remove it when no longer
needed:

```bash
kind delete cluster --name vehicle-classifier-phase6
```

Deleting this kind cluster removes its ephemeral Prometheus and Grafana data;
it does not delete Git files, DVC data, or Docker images.
