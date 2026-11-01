# Phase 8 evidence index

Evidence is limited to checks actually run in this workspace. Screenshots of
external GitHub Actions or Jenkins screens are intentionally absent because
those systems were not connected/configured here. Do not represent command
output or a local dashboard as evidence of a remote deployment.

| Evidence | Source/status |
|---|---|
| Target and OOD metrics/confusion matrices | `../../reports/evaluation.json`, `../../reports/ood_evaluation.json` |
| MLflow run table screenshot | [`mlflow-local-runs.png`](mlflow-local-runs.png); the local `VehicleClassifier` experiment had 9 matching runs and includes the two fresh Phase 8 evaluations |
| MLflow model registry screenshot | [`mlflow-model-versions.png`](mlflow-model-versions.png); versions 1 and 2 shown, with version 2 carrying the `champion` alias |
| Grafana dashboard screenshot | [`grafana-local-kind.png`](grafana-local-kind.png); captured from the local kind dashboard after Phase 8 smoke traffic |
| DVC pipeline and manifest | `../../dvc.yaml`, `../../dvc.lock`, `../../data/manifests/data_manifest.json` |
| Champion package identity | `../../student_model.pth.json`; validated by `scripts/create_release_metadata.py --validate-only` |
| Container offline API smoke | Phase 8 local Docker run: passed live/ready, valid prediction, malformed image rejection with `--network none` |
| Kubernetes rollout/failure/recovery | [`kubernetes-local-rollout.txt`](kubernetes-local-rollout.txt); successful local rollout, intentionally unavailable image, rollback and recovered prediction |
| Monitoring | Existing local Prometheus/Grafana in kind; see `docs/phase7_monitoring.md` and `reports/phase7_local_load.json` |
| GitHub Actions screenshot/release | Pending: workflow must run in the user's GitHub repository |
| Jenkins screenshot/digest rollout | Pending: Jenkins agent/credentials and GHCR immutable digest required |

The local clean-clone DVC retrieval was not completed in this Phase 8 pass:
the second copy exceeded this machine's temporary storage quota. The previously
recorded Phase 2 retrieval evidence remains documented in `docs/phase_status.md`;
repeat it on a machine with sufficient free storage to reconfirm.
