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
| 4 — Inference API/container | FastAPI upload endpoint, response/error schemas, request/file/pixel limits, health probes, startup package validation, CPU Docker image, and local/container smoke scripts are implemented. | 12 tests pass locally. Docker image built successfully; `docker run --network none` passed ready/live, valid prediction, and malformed-image checks; host port mapping also passed the same smoke checks. Runtime inference was verified with network access disabled. |

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

- DVC collaboration has not been verified from a friend's machine. Current DVC storage is the local RustFS service; sharing Git code alone does not share its objects.
- The two MLflow training comparisons are short smoke runs, not full training experiments. They prove that configured runs can be tracked, but do not support a meaningful hyperparameter or model-quality comparison. Full experiments are deferred; run at least two complete, reproducible configurations before claiming the full Phase 3 experiment-comparison gate is complete.
- Phase 0 team ratification and dataset source/redistribution provenance remain open.
