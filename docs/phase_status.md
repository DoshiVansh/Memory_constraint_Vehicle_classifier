# MLOps phase status

Updated: 2026-10-25

This records implementation state separately from exit-gate evidence. A checked
implementation item does not mean its end-to-end gate has passed.

| Phase | Implementation state | Exit-gate evidence |
|---|---|---|
| 0 — Baseline audit | Audit and provisional contracts documented in `docs/baseline_audit.md`. | Pending: fresh baseline, confusion matrix, verified data provenance/version, team ratification. |
| 1 — Cleanup/configuration | Core modules and YAML configs are present; legacy `VehicleClassifier.predict(image_path) -> int` remains. | Not re-run in this environment. |
| 2 — DVC/data versioning | DVC manifest/validation stages, dataset ignore rules, local RustFS setup, and training/evaluation stages are defined. | Pending: Docker access was rejected; DVC dataset pointers, remote storage, manifest validation, and push/pull are not verified. |
| 3 — Training/MLflow/promotion | Training and evaluation tracking, DVC pipeline links, strict baseline/size gate, candidate registration, and explicit alias promotion are implemented. | Pending: install MLflow, run two experiments, inspect UI, refresh baseline, pass the gate, and register/promote a candidate. |

## Current promotion contract

- A candidate must have a tracked evaluation from a clean committed revision, an MLflow run ID, the same dataset-manifest checksum as the baseline, and a checkpoint hash matching the evaluated checkpoint.
- The canonical baseline must contain fresh metrics. Historical 92.44% results in `reports/baseline.json` are not accepted as a promotion threshold.
- The serialized checkpoint must be strictly smaller than 5,000,000 bytes.
- Registration creates a version tagged `approval_status=candidate`. Only `scripts/promote_model.py --approve` may assign the `champion` alias, and only for a gate-passing candidate.

## Storage scope

MLflow uses a local SQLite metadata store and local artifact directory by default;
this is free and suitable for a local demo. It is not shared across machines by
itself. DVC's local RustFS remote is also not running or populated yet; see the
Phase 2 setup and limitations in `README.md`.
