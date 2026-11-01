# System architecture

```mermaid
flowchart LR
  subgraph Data[Versioned data]
    SRC[Curated image folders] --> DVC[DVC manifest and pipeline]
    DVC --> OBJ[Local RustFS object store]
    DVC --> VAL[Data validation report]
  end
  subgraph Train[Model development]
    CFG[YAML configs] --> TRAIN[Teacher and student training]
    VAL --> TRAIN
    TRAIN --> CKPT[Student checkpoint and sidecar]
    TRAIN --> MLF[MLflow runs and model registry]
    CKPT --> EVAL[Target and OOD evaluation]
    EVAL --> GATE[Size and quality gate]
    GATE --> MLF
    MLF --> PROM[Approved champion alias]
  end
  subgraph Serve[Serving and operations]
    CKPT --> IMG[CPU inference Docker image]
    PROM -. identity/version .-> IMG
    IMG --> API[FastAPI: predict, health, metrics]
    API --> K8S[Kubernetes Deployment and Service]
    K8S --> PR[Prometheus scrape and alerts]
    PR --> GF[Grafana dashboard]
    CI[GitHub Actions quality/release workflows] -. build/publish .-> IMG
    J[Jenkins digest deployment pipeline] -. deploy/verify/rollback .-> K8S
  end
```

## Data and control flow

Git stores code, small configuration, DVC pointers, Kubernetes manifests, and
documentation. DVC stores checksummed dataset/model objects in the configured
object store; this project currently uses a free local RustFS instance. DVC
stage hashes express dependencies and outputs. MLflow stores run parameters,
metrics, lineage, and model versions in the local SQLite/artifact store.

The inference container includes the approved checkpoint and integrity sidecar,
so runtime prediction does not download weights. The API serves CPU inference,
reports bounded-label request and prediction metrics, and is deployed behind a
ClusterIP. Prometheus scrapes those metrics; Grafana reads Prometheus. GitHub
Actions and Jenkins are implemented as automation definitions, but remote
GitHub publication and Jenkins deployment remain pending external setup.
