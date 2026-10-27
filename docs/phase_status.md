# MLOps phase status

Updated: 2026-10-26

This records implementation state separately from exit-gate evidence. A checked
implementation item does not mean its end-to-end gate has passed.

| Phase | Implementation state | Exit-gate evidence |
|---|---|---|
| 0 — Baseline audit | Audit and provisional contracts documented in `docs/baseline_audit.md`; a fresh run now exists. | Fresh target run, confusion matrix, per-class results, checkpoint hash, manifest hash, and CPU latency recorded. Team ratification and source provenance remain open. |
| 1 — Cleanup/configuration | Core modules and YAML configs are present; legacy `VehicleClassifier.predict(image_path) -> int` remains. | `pytest -q`: 9 passed in the available Python 3.10 environment. |
| 2 — DVC/data versioning | DVC pointers, manifest/validation stages, local RustFS bucket, and training/evaluation stages are defined. Curated dataset objects and manifest cache were pushed. | Passed locally: temporary checkout retrieved all 24,690 dataset files, rebuilt the manifest with the same SHA-256, and reproduced the validation report with zero errors. Review warnings remain. Raw dataset redistribution rights remain unverified, so raw dumps stay local. |
| 3 — Training/MLflow/promotion | MLflow tracking, DVC pipeline links, bounded student training demonstrations, evaluation packaging, quality gate, candidate registration, and explicit alias promotion are implemented. | Fresh baseline recorded; DVC target/OOD evaluation and registered-model loading verified; two short smoke runs plus evaluations appear in MLflow; candidate version 2 passed the baseline/size gate and is linked to the final DVC evaluation run. Version 1 is marked superseded. Version 2 is approved and verified as the `champion` alias. Full training experiments remain deferred. |
| 4 — Inference API/container | FastAPI upload endpoint, response/error schemas, request/file/pixel limits, health probes, startup package validation, CPU Docker image, and local/container smoke scripts are implemented. | 12 API/legacy tests passed at Phase 4 completion. The current full suite has 18 passing tests. Docker image built successfully; offline and host port-mapped prediction/rejection smoke checks passed. |
| 5 — CI and immutable image build | PR quality workflow, reusable CI gate, manually triggered self-hosted GPU training workflow, tag-triggered GHCR publishing, and release lineage manifest are implemented. | Local Ruff, Python compilation, YAML parsing, and pytest pass (18 tests). Workflows have not yet run in GitHub; branch protection, a first successful GHCR release, and self-hosted GPU runner/DVC credentials still need GitHub-side setup. |
| 6 — Kubernetes deployment and release safety | Local staging Namespace, ConfigMap/Deployment/Service templates, bounded CPU inference pod with probes, digest-only Jenkins pipeline, metadata validator, readiness/prediction smoke script, and rollback runbook are implemented. | Demonstrated on disposable kind cluster: local Phase 4 image rolled out; readiness and PNG prediction passed; unavailable-image rollout failed; revision 1 was restored and both smoke checks passed. Jenkins automation and published immutable GHCR digest remain unverified, so the full gate is still open. |
| 7 — Monitoring and performance | Bounded-label Prometheus API metrics, in-cluster Prometheus scrape/alert rules, provisioned eight-panel Grafana dashboard, repeatable concurrency load script, and runbook are implemented. | Demonstrated in local kind: target UP, alert group loaded, Grafana datasource healthy/dashboard loaded, and 300/300 synthetic predictions succeeded across concurrency 1/2/4. p50/p95, throughput, error rate, and sampled process CPU/RSS are recorded in `reports/phase7_local_load.json`. Local-only result; no production SLO or notification integration claimed. |
| 8 — End-to-end hardening and presentation | Architecture, model card, operator runbook, demo/FAQ, evidence index, and final README command path added. MLflow and Grafana screenshots and a Kubernetes rollout transcript are in `docs/evidence/`. | Existing champion evaluated via DVC single stages and MLflow; target 92.40%, mapped OOD 43.24%. Champion package contract printed valid identity (2,665,366 bytes; expected SHA). Docker image built; offline container smoke passed. Local kind rollout succeeded; readiness and real prediction returned matching champion SHA. An intentionally unavailable image timed out, rollback restored the prior revision, and smoke passed again. Prometheus/Grafana stayed available. Clean-clone DVC retrieval was blocked on repeat by temporary storage quota; prior Phase 2 retrieval remains recorded separately. GitHub Actions/release and Jenkins runtime need external setup, and no GHCR tag was created. Full clean-clone exit gate is not yet closed. |

## Current promotion contract

- A candidate must have a tracked evaluation with a Git SHA and archived source snapshot fingerprint, an MLflow run ID, the same dataset-manifest checksum as the baseline, and a checkpoint hash matching the evaluated checkpoint.
- The canonical baseline must contain fresh metrics. Historical 92.44% results in `reports/baseline.json` are not accepted as a promotion threshold.
- The serialized checkpoint must be strictly smaller than 5,000,000 bytes.
- Registration creates a version tagged `approval_status=candidate`. Only `scripts/promote_model.py --approve` may assign the `champion` alias, and only for a gate-passing candidate. Current local MLflow state: version 2 is `champion`, tagged `approval_status=approved` and `quality_gate=passed`.

## Storage scope

MLflow uses a local SQLite metadata store and local artifact directory by default;
this is free and suitable for a local demo. It is not shared across machines by
itself. RustFS is running locally on ports 9000/9001 and DVC data is in its local
bucket. Keep both services running to use the web consoles; neither is exposed
to the public internet.

The validation/training runs used the available Python 3.10 environment with
PyTorch 2.5.1/torchvision 0.20.1, not the repository's pinned PyTorch 2.6.0 /
torchvision 0.21.0. CUDA was unavailable to this runtime, so the two short
tracking demonstrations ran on CPU. Full teacher/student training is still a
long-running reproducibility stage and was not run end-to-end.

## Deferred exit-gate work

- Phase 8: Repeat `dvc pull` from a clean clone on a computer with enough free disk, then follow `docs/runbook.md`. Capture real GitHub Actions and Jenkins/GHCR screenshots only after those external workflows are configured. The local presentation package is ready, but the clean-clone/external automation portion of the exit gate remains open.

- DVC collaboration has not been verified from a friend's machine. Current DVC storage is the local RustFS service; sharing Git code alone does not share its objects.
- The two MLflow training comparisons are short smoke runs, not full training experiments. They prove that configured runs can be tracked, but do not support a meaningful hyperparameter or model-quality comparison. Full experiments are deferred; run at least two complete, reproducible configurations before claiming the full Phase 3 experiment-comparison gate is complete.
- Phase 0 team ratification and dataset source/redistribution provenance remain open.
- Phase 5 merge blocking requires the repository to mark `PR Quality Gate / test-and-static-checks` as a required status check. The GPU training workflow additionally needs a runner labeled `gpu` plus DVC endpoint/access-key settings. A release tag should be pushed only after the commit's PR gate passes.
- Phase 6 still requires a configured Jenkins agent labeled `kubernetes-deployer`, Jenkins secret-file credential `vehicle-classifier-kubeconfig`, and a published GHCR image digest. The local Kubernetes success/failure/rollback was demonstrated with the local Phase 4 image; capture a Jenkins deployment using the GHCR digest before closing the full gate.
- Phase 7 is demonstrated locally. Before setting SLOs/production alerts, repeat the load test on the actual deployment hardware/network. Add Alertmanager/Grafana contact points only if notification delivery is needed; no contact credentials or destination were supplied.
