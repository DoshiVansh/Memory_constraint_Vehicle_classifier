# 3–5 minute project demo and interview FAQ

## Suggested 4-minute walkthrough

1. **0:00–0:35 — Problem and boundary.** Show README and model card. Explain
   the five labels, the 2.67 MB student checkpoint, CPU inference, and that
   this is a student demonstration rather than safety-critical software.
2. **0:35–1:15 — Data lineage.** Show `data/manifests/data_manifest.json`,
   `dvc.yaml`, and MLflow. Explain Git stores pointers/config while DVC stores
   checksummed data objects; MLflow records run history and model versions.
3. **1:15–2:05 — Evaluation and promotion.** Open the target/OOD JSON and
   model card. Report 92.4% target and 43.24% mapped OOD accuracy, and explain
   why the OOD result is an important limitation. Show champion version 2 and
   its checkpoint SHA rather than just a filename.
4. **2:05–3:00 — Serving.** Show FastAPI `/docs`, `/health/ready`, and one image
   prediction; mention upload limits and that the image has no runtime weight
   download. Show the successful network-disabled container smoke test.
5. **3:00–4:00 — Operations and honest status.** Show the kind rollout, the
   Prometheus target and Grafana dashboard. Explain the previously demonstrated
   failed-image rollout/rollback. State GitHub release and Jenkins execution
   need external setup and were not claimed as completed here.

Use the exact commands in `README.md` and [`runbook.md`](runbook.md). Do not
present locally captured evidence as GitHub Actions or Jenkins evidence.

## Short technical FAQ

**Why DVC and MLflow?** DVC tracks data/stage dependencies and content hashes;
MLflow captures experiment metrics, artifacts, model versions and approval.
They answer different lineage questions.

**Where is DVC hosted?** On this machine's free local RustFS service. It is
not a public server and another teammate cannot pull until the object store is
reachable and credentials are shared securely.

**Does training use the GPU?** Configs select CUDA when a CUDA-enabled PyTorch
build and visible compatible GPU exist. The inference container and recorded
evaluations use CPU. Do not infer GPU training from an API deployment.

**Why is OOD performance low?** MIO-TCD differs from the curated target data;
class semantics are imperfectly mapped, especially `None`. The 43.24% result
shows the model's limits under this stress test, not a robust deployment score.

**What exactly is champion?** MLflow registry alias `champion` points to
version 2 locally; the sidecar binds that identity to the checkpoint SHA and
preprocessing/class contract. The alias is in local MLflow state, not synced
to GitHub.

**What remains outside the local demo?** GitHub branch protection/CI and
GHCR release evidence, a configured Jenkins agent and digest deployment, and
verification that a teammate can retrieve from DVC. These require account,
runner, storage/network, and credential setup.
