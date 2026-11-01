"""Generate rich Grafana dashboard JSON and update k8s/monitoring/grafana.yaml."""

import json
from pathlib import Path


def generate_dashboard() -> dict:
    return {
        "id": None,
        "uid": "vehicle-classifier-overview",
        "title": "Vehicle Classifier Enterprise MLOps Observability",
        "description": "Real-time production monitoring: inference traffic, latency percentiles, error rates, model class distribution, pod resources, and OpenTelemetry distributed tracing.",
        "schemaVersion": 41,
        "version": 2,
        "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "timezone": "browser",
        "panels": [
            # --- ROW 1: EXECUTIVE KPIS ---
            {
                "id": 1,
                "type": "stat",
                "title": "Total Inferences Served",
                "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "sum(vehicle_classifier_requests_total{endpoint=\"/predict\"})", "instant": True}],
                "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "unit": "short"}},
                "options": {"reduceOptions": {"calcs": ["lastNotNull"], "values": False}, "graphMode": "none", "textMode": "auto"}
            },
            {
                "id": 2,
                "type": "stat",
                "title": "Prediction Throughput",
                "gridPos": {"h": 4, "w": 4, "x": 4, "y": 0},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "sum(rate(vehicle_classifier_requests_total{endpoint=\"/predict\"}[1m]))", "instant": True}],
                "fieldConfig": {
                    "defaults": {
                        "unit": "reqps",
                        "thresholds": {"mode": "absolute", "steps": [{"color": "blue", "value": None}, {"color": "green", "value": 0.5}, {"color": "yellow", "value": 10}]}
                    }
                },
                "options": {"reduceOptions": {"calcs": ["lastNotNull"], "values": False}, "graphMode": "none", "textMode": "auto"}
            },
            {
                "id": 3,
                "type": "stat",
                "title": "p95 Latency SLA",
                "gridPos": {"h": 4, "w": 4, "x": 8, "y": 0},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "1000 * histogram_quantile(0.95, sum by (le) (rate(vehicle_classifier_request_duration_seconds_bucket{endpoint=\"/predict\"}[2m])))", "instant": True}],
                "fieldConfig": {
                    "defaults": {
                        "unit": "ms",
                        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 25}, {"color": "red", "value": 35}]}
                    }
                },
                "options": {"reduceOptions": {"calcs": ["lastNotNull"], "values": False}, "graphMode": "none", "textMode": "auto"}
            },
            {
                "id": 4,
                "type": "stat",
                "title": "HTTP Success Rate",
                "gridPos": {"h": 4, "w": 4, "x": 12, "y": 0},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": "(1 - (sum(rate(vehicle_classifier_errors_total{endpoint=\"/predict\"}[5m])) or vector(0)) / clamp_min(sum(rate(vehicle_classifier_requests_total{endpoint=\"/predict\"}[5m])), 0.001)) * 100",
                        "instant": True
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "thresholds": {"mode": "absolute", "steps": [{"color": "red", "value": None}, {"color": "yellow", "value": 90}, {"color": "green", "value": 95}]}
                    }
                },
                "options": {"reduceOptions": {"calcs": ["lastNotNull"], "values": False}, "graphMode": "none", "textMode": "auto"}
            },
            {
                "id": 5,
                "type": "stat",
                "title": "Active API Replicas",
                "gridPos": {"h": 4, "w": 4, "x": 16, "y": 0},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "count(up{job=\"vehicle-classifier-api\"} == 1)", "instant": True}],
                "fieldConfig": {"defaults": {"thresholds": {"mode": "absolute", "steps": [{"color": "yellow", "value": None}, {"color": "green", "value": 2}]}}},
                "options": {"reduceOptions": {"calcs": ["lastNotNull"], "values": False}, "graphMode": "none", "textMode": "auto"}
            },
            {
                "id": 6,
                "type": "stat",
                "title": "Champion Model",
                "gridPos": {"h": 4, "w": 4, "x": 20, "y": 0},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "count(vehicle_classifier_model_info)", "instant": True}],
                "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "purple", "value": None}]}}},
                "options": {"reduceOptions": {"calcs": ["lastNotNull"], "values": False}, "graphMode": "none", "textMode": "auto"}
            },

            # --- ROW 2: TRAFFIC & PREDICTIONS BREAKDOWN ---
            {
                "id": 7,
                "type": "timeseries",
                "title": "Prediction Request Rate (Throughput)",
                "gridPos": {"h": 8, "w": 10, "x": 0, "y": 4},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "sum(rate(vehicle_classifier_requests_total{endpoint=\"/predict\"}[1m]))", "legendFormat": "Total req/sec"}],
                "fieldConfig": {"defaults": {"unit": "reqps", "custom": {"fillOpacity": 30, "gradientMode": "opacity", "lineWidth": 2}}},
                "options": {"legend": {"showLegend": True, "displayMode": "table", "calcs": ["mean", "max", "lastNotNull"]}}
            },
            {
                "id": 8,
                "type": "piechart",
                "title": "Vehicle Class Predictions Distribution",
                "gridPos": {"h": 8, "w": 7, "x": 10, "y": 4},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [
                    {
                        "refId": "A",
                        "expr": "sum by (predicted_class) (increase(vehicle_classifier_predictions_total[15m]))",
                        "legendFormat": "{{predicted_class}}",
                        "instant": True
                    }
                ],
                "fieldConfig": {"defaults": {"unit": "short"}},
                "options": {
                    "pieType": "donut",
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                    "legend": {"displayMode": "table", "values": ["percent", "value"]}
                }
            },
            {
                "id": 9,
                "type": "timeseries",
                "title": "Predictions by Class Over Time",
                "gridPos": {"h": 8, "w": 7, "x": 17, "y": 4},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "sum by (predicted_class) (rate(vehicle_classifier_predictions_total[1m]))", "legendFormat": "{{predicted_class}}"}],
                "fieldConfig": {"defaults": {"unit": "reqps", "custom": {"stacking": {"mode": "normal", "group": "A"}, "fillOpacity": 40}}},
                "options": {"legend": {"showLegend": True, "displayMode": "list"}}
            },

            # --- ROW 3: LATENCY & SLA PERFORMANCE ---
            {
                "id": 10,
                "type": "timeseries",
                "title": "End-to-End Request Turnaround Latency (p50 / p90 / p95 / p99)",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [
                    {"refId": "A", "expr": "1000 * histogram_quantile(0.50, sum by (le) (rate(vehicle_classifier_request_duration_seconds_bucket{endpoint=\"/predict\"}[2m])))", "legendFormat": "p50 Median"},
                    {"refId": "B", "expr": "1000 * histogram_quantile(0.90, sum by (le) (rate(vehicle_classifier_request_duration_seconds_bucket{endpoint=\"/predict\"}[2m])))", "legendFormat": "p90"},
                    {"refId": "C", "expr": "1000 * histogram_quantile(0.95, sum by (le) (rate(vehicle_classifier_request_duration_seconds_bucket{endpoint=\"/predict\"}[2m])))", "legendFormat": "p95 SLA (Target <25ms)"},
                    {"refId": "D", "expr": "1000 * histogram_quantile(0.99, sum by (le) (rate(vehicle_classifier_request_duration_seconds_bucket{endpoint=\"/predict\"}[2m])))", "legendFormat": "p99 Tail"}
                ],
                "fieldConfig": {"defaults": {"unit": "ms", "custom": {"lineWidth": 2, "spanNulls": True}}},
                "options": {"legend": {"showLegend": True, "displayMode": "table", "calcs": ["mean", "max", "lastNotNull"]}}
            },
            {
                "id": 11,
                "type": "timeseries",
                "title": "Pure Model Inference vs API Network Overhead",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [
                    {"refId": "A", "expr": "1000 * histogram_quantile(0.95, sum by (le) (rate(vehicle_classifier_inference_duration_seconds_bucket[2m])))", "legendFormat": "p95 Pure PyTorch Forward Pass"},
                    {"refId": "B", "expr": "1000 * histogram_quantile(0.95, sum by (le) (rate(vehicle_classifier_request_duration_seconds_bucket{endpoint=\"/predict\"}[2m])))", "legendFormat": "p95 Total Turnaround (Decode + Preprocess + Net)"}
                ],
                "fieldConfig": {"defaults": {"unit": "ms", "custom": {"lineWidth": 2, "fillOpacity": 15}}},
                "options": {"legend": {"showLegend": True, "displayMode": "table", "calcs": ["mean", "max", "lastNotNull"]}}
            },

            # --- ROW 4: INFRASTRUCTURE & CONTAINER HEALTH ---
            {
                "id": 12,
                "type": "timeseries",
                "title": "API Container CPU Usage",
                "gridPos": {"h": 7, "w": 6, "x": 0, "y": 20},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "sum(rate(process_cpu_seconds_total{job=\"vehicle-classifier-api\"}[1m]))", "legendFormat": "Total CPU Cores"}],
                "fieldConfig": {"defaults": {"unit": "cores", "custom": {"fillOpacity": 20}}},
                "options": {"legend": {"showLegend": True}}
            },
            {
                "id": 13,
                "type": "timeseries",
                "title": "API Resident Memory (RAM RSS)",
                "gridPos": {"h": 7, "w": 6, "x": 6, "y": 20},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "sum(process_resident_memory_bytes{job=\"vehicle-classifier-api\"})", "legendFormat": "Total Memory"}],
                "fieldConfig": {"defaults": {"unit": "bytes", "custom": {"fillOpacity": 20}}},
                "options": {"legend": {"showLegend": True}}
            },
            {
                "id": 14,
                "type": "gauge",
                "title": "Active In-Flight Requests",
                "gridPos": {"h": 7, "w": 6, "x": 12, "y": 20},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "sum(vehicle_classifier_in_progress_requests) or vector(0)", "instant": True}],
                "fieldConfig": {
                    "defaults": {
                        "unit": "short",
                        "min": 0,
                        "max": 10,
                        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 4}, {"color": "red", "value": 8}]}
                    }
                },
                "options": {"showThresholdLabels": False, "showThresholdMarkers": True}
            },
            {
                "id": 15,
                "type": "timeseries",
                "title": "HTTP Status Distribution (200 vs 4xx vs 5xx)",
                "gridPos": {"h": 7, "w": 6, "x": 18, "y": 20},
                "datasource": {"type": "prometheus", "uid": "vehicle-classifier-prometheus"},
                "targets": [{"refId": "A", "expr": "sum by (status) (rate(vehicle_classifier_requests_total[1m]))", "legendFormat": "HTTP {{status}}"}],
                "fieldConfig": {"defaults": {"unit": "reqps", "custom": {"stacking": {"mode": "normal"}}}},
                "options": {"legend": {"showLegend": True}}
            },

            # --- ROW 5: OPENTELEMETRY DISTRIBUTED TRACING ---
            {
                "id": 16,
                "type": "text",
                "title": "🔬 OpenTelemetry Distributed Tracing & Execution Waterfall",
                "gridPos": {"h": 9, "w": 24, "x": 0, "y": 27},
                "options": {
                    "mode": "markdown",
                    "content": (
                        "# 🔬 Live Distributed Tracing & Span Waterfall Dashboard\n\n"
                        "To select individual request threads and inspect the exact API calls, child functions, microsecond latencies, and span attributes:\n\n"
                        "### 🚀 [Open Interactive Tracing Waterfall UI (http://20.200.237.154:8000/traces)](http://20.200.237.154:8000/traces)\n"
                        "**JSON Traces Endpoint**: [`/api/traces`](http://20.200.237.154:8000/api/traces) | "
                        "**Interactive Swagger Docs**: [`/docs`](http://20.200.237.154:8000/docs)\n\n"
                        "---\n\n"
                        "### 📊 Execution Waterfall & Function Call Hierarchy\n"
                        "```\n"
                        "predict_request (HTTP POST /predict - Root Span)\n"
                        "  ├── [01] read_upload_payload          : validates multipart byte streams, buffers memory\n"
                        "  ├── [02] validate_and_decode_image    : image format safety checks, PIL decompression\n"
                        "  ├── [03] convert_image_rgb            : transforms RGBA/Grayscale to RGB tensor\n"
                        "  └── [04] model_inference              : normalizes tensor & executes neural model\n"
                        "        └── [05] pytorch_forward_pass   : ShuffleNet V2 forward graph computation\n"
                        "```\n\n"
                        "#### ✨ Features of the Traces Dashboard:\n"
                        "- **Thread / Request Selector**: Click any request on the left sidebar to load its complete trace.\n"
                        "- **Gantt Waterfall Timeline**: Visually inspect relative timing and parallel overhead of every sub-function.\n"
                        "- **Span Inspector Card**: Click any span to see microsecond duration, parent/child IDs, `image.format`, `image.size_bytes`, `prediction.class`, `prediction.confidence`, and `thread.id`.\n"
                        "- **Real-time Live Stream**: Auto-polls every 3 seconds to capture fresh traffic from the in-cluster traffic generator."
                    )
                }
            }
        ]
    }


def update_manifest():
    dashboard = generate_dashboard()
    dash_json = json.dumps(dashboard, indent=2)

    # Indent by 4 spaces for the YAML literal block
    indented_json = "\n".join("    " + line if line.strip() else "" for line in dash_json.splitlines())

    manifest_template = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: vehicle-classifier-grafana-provisioning
  namespace: vehicle-classifier-staging
data:
  datasources.yaml: |
    apiVersion: 1
    datasources:
      - name: Prometheus
        uid: vehicle-classifier-prometheus
        type: prometheus
        access: proxy
        url: http://prometheus.vehicle-classifier-staging.svc.cluster.local:9090
        isDefault: true
        editable: false
  dashboards.yaml: |
    apiVersion: 1
    providers:
      - name: vehicle-classifier
        orgId: 1
        folder: Vehicle Classifier
        type: file
        disableDeletion: true
        editable: false
        options:
          path: /etc/grafana/dashboards
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: vehicle-classifier-grafana-dashboard
  namespace: vehicle-classifier-staging
data:
  vehicle-classifier.json: |
{indented_json}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vehicle-classifier-grafana
  namespace: vehicle-classifier-staging
  labels:
    app.kubernetes.io/name: vehicle-classifier-grafana
    app.kubernetes.io/part-of: vehicle-classifier
spec:
  replicas: 1
  revisionHistoryLimit: 5
  strategy:
    type: RollingUpdate
  selector:
    matchLabels:
      app.kubernetes.io/name: vehicle-classifier-grafana
  template:
    metadata:
      labels:
        app.kubernetes.io/name: vehicle-classifier-grafana
        app.kubernetes.io/part-of: vehicle-classifier
    spec:
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 472
        fsGroup: 472
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: grafana
          image: grafana/grafana:13.2.4
          imagePullPolicy: IfNotPresent
          env:
            - name: GF_SECURITY_ADMIN_PASSWORD
              value: "admin"
            - name: GF_USERS_ALLOW_SIGN_UP
              value: "false"
            - name: GF_AUTH_ANONYMOUS_ENABLED
              value: "true"
            - name: GF_AUTH_ANONYMOUS_ORG_ROLE
              value: "Viewer"
            - name: GF_AUTH_BASIC_ENABLED
              value: "true"
            - name: GF_LOG_LEVEL
              value: "warn"
          ports:
            - name: http
              containerPort: 3000
          readinessProbe:
            httpGet:
              path: /api/health
              port: http
            periodSeconds: 5
            failureThreshold: 12
          livenessProbe:
            httpGet:
              path: /api/health
              port: http
            periodSeconds: 10
          resources:
            requests:
              cpu: 25m
              memory: 128Mi
            limits:
              cpu: 250m
              memory: 256Mi
          volumeMounts:
            - name: provisioning
              mountPath: /etc/grafana/provisioning/datasources/datasources.yaml
              subPath: datasources.yaml
            - name: provisioning
              mountPath: /etc/grafana/provisioning/dashboards/dashboards.yaml
              subPath: dashboards.yaml
            - name: dashboards
              mountPath: /etc/grafana/dashboards/vehicle-classifier.json
              subPath: vehicle-classifier.json
            - name: data
              mountPath: /var/lib/grafana
      volumes:
        - name: provisioning
          configMap:
            name: vehicle-classifier-grafana-provisioning
        - name: dashboards
          configMap:
            name: vehicle-classifier-grafana-dashboard
        - name: data
          emptyDir:
            sizeLimit: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: vehicle-classifier-staging
  labels:
    app.kubernetes.io/name: vehicle-classifier-grafana
spec:
  type: LoadBalancer
  selector:
    app.kubernetes.io/name: vehicle-classifier-grafana
  ports:
    - name: http
      port: 3000
      targetPort: http
"""
    Path("k8s/monitoring/grafana.yaml").write_text(manifest_template, encoding="utf-8")
    print("Successfully generated upgraded k8s/monitoring/grafana.yaml with 16 panels!")


if __name__ == "__main__":
    update_manifest()
