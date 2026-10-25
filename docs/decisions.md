# Architecture decision records

## ADR-001: Canonical student architecture

- **Status:** Provisional; ratify after fresh baseline reproduction.
- **Decision:** Use torchvision ShuffleNet V2 x1.0 with five output classes as the intended canonical architecture. Candidate artifact: `student_model.pth`, SHA-256 `c80eaf7a45e87e604b2e41af992094425834e46d747315a6fb66a65f7c18e1d4`.
- **Reason:** It agrees with the serving implementation, distillation intent, report, and historical 92.44% checkpoint (the logged filename is byte-identical to the candidate).
- **Consequences:** Do not use x0.5 or SqueezeNet metrics to characterize this artifact. The current artifact is under 5,000,000 bytes. Fresh reproduction remains outstanding.

## ADR-002: Class order and preprocessing contract

- **Status:** Provisional; ratify after baseline run.
- **Decision:** Output indices are `0 Bus`, `1 Truck`, `2 Car`, `3 Bike`, `4 None`. Inference converts images to RGB, resizes to 224x224, converts to tensor, then applies ImageNet mean/std normalization.
- **Reason:** This is the mapping declared in `vehicle_classifier.py`; training data directory prefixes produce the same order, and the legacy evaluator maps plain test folders explicitly.
- **Consequences:** Any new data loader or API must apply this mapping explicitly rather than relying on unprefixed `ImageFolder` ordering.

## ADR-003: Offline checkpoint loading

- **Status:** Accepted.
- **Decision:** Construct the serving architecture with `weights=None` and load the complete trained checkpoint state dict.
- **Reason:** The full checkpoint replaces initialization weights; a pretrained-weight download is unnecessary and undermines offline startup.
- **Consequences:** A checkpoint is required for meaningful inference. Validate the load on a PyTorch-capable, network-restricted environment when available.

## ADR-004: Provisional target dataset and baseline schema

- **Status:** Provisional; dataset provenance and leakage are unresolved.
- **Decision:** Use the current 2,500-image `test_dataset/` as the evaluation candidate. Record its content fingerprint and emit classwise metrics plus a 5x5 confusion matrix indexed in canonical class order. Preserve historical results separately from fresh-run results.
- **Reason:** It is locally available and the historical evaluator reports 500 samples per class, but no versioned manifest identifies provenance.
- **Consequences:** Historical accuracy is not a quality gate. Do not ratify data version or promotion thresholds until a fresh run and dataset audit are completed.

## ADR-005: Local MLflow tracking and approved model alias

- **Status:** Accepted for the local MVP; shared-server deployment remains a later decision.
- **Decision:** Use MLflow Tracking with a local SQLite backend and local artifact directory by default. Log each teacher, student, and evaluation run. Register the evaluation's packaged PyFunc model only after the baseline and strict artifact-size gates pass; require a separate explicit approval command to assign the `champion` alias.
- **Reason:** SQLite supports the Model Registry without a paid service, while versioned model artifacts, run IDs, tags, and aliases make experiments traceable on a student laptop.
- **Consequences:** The default database and artifacts are machine-local and ignored by Git. Teammates do not automatically share experiment history. DVC remains responsible for data/pipeline revisions; MLflow records the run and model lineage. Historical accuracy alone cannot authorize promotion.
