# Phase 0 baseline audit

**Audit date:** 2026-10-26

**Repository revision evaluated:** `0762ae53911e71a0ac2058073522d43f7ff9f225`

**Status:** A fresh target evaluation, confusion matrix, and DVC manifest are recorded. The source provenance and team ratification remain open.

## Selected baseline candidate

- **Architecture:** torchvision ShuffleNet V2 x1.0, five output logits (`model.fc`). This agrees with `vehicle_classifier.py` and the distillation script's intended student.
- **Checkpoint:** `student_model.pth`, 2,665,366 bytes (2.54 MiB; 2.67 decimal MB), SHA-256 `c80eaf7a45e87e604b2e41af992094425834e46d747315a6fb66a65f7c18e1d4`.
- **Checkpoint identity:** The untracked `shuffleNet_v2_model.pth` has the same SHA-256, so it is byte-identical to the selected checkpoint. The historical `model_test_log.txt` reports 92.44% on 2,500 images for that filename. This ties the historical score to these exact bytes, but it does not establish that the current dataset contents are identical to those used for that historical run.
- **Dataset candidate:** `test_dataset/`, 2,500 images: 500 each in `Bus`, `Truck`, `Car`, `Bike`, and `None`. The current test tree aggregate (sorted per-file SHA-256 list hashed as bytes) is `34ce9044e7b87f67c45208254470486e59680518e0f2bfd0c787876e1d20c4ac`.
- **Fresh data identity:** Manifest SHA-256 `3c40f2a7c554220c35c98db4b299a29b3f9c6391f2d56a31695796bc8ccfd35a`; dataset manifest and validation report were rebuilt through DVC. Exact cross-split duplicate leakage is zero after grouping identical training files into one split. Source lineage and near-duplicate warnings remain for review.
- **Class contract:** `{0: Bus, 1: Truck, 2: Car, 3: Bike, 4: None}`. Training folders are prefixed `0_Bus`, `1_Truck`, `2_Car`, `3_Bike`, `4_None`; this makes `ImageFolder`'s alphabetical order agree with the contract. Evaluation folders are unprefixed and the legacy evaluator explicitly maps them to the same indices.
- **Preprocessing in serving:** RGB conversion; resize directly to 224x224; tensor conversion; ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Distillation validation uses the same resize and normalization. Training adds random augmentation before normalization.
- **Serialized size gate:** The selected artifact is below the strict 5,000,000-byte limit. This is file size only, not peak process/container RAM.

## Findings and inconsistencies

1. The serving model factory previously requested `ShuffleNet_V2_X1_0_Weights.DEFAULT` even though the full checkpoint is loaded afterward. That could fetch an unnecessary pretrained file and prevented reliable offline inference. It now constructs with `weights=None`; loading the full trained state remains required when a checkpoint is supplied.
2. Several checkpoints and historical logs coexist. The 92.44% log entry is for `shuffleNet_v2_model.pth`, which is byte-identical to `student_model.pth`; the smaller x0.5 model has a separate historical score of 91.28%. Do not mix them.
3. The legacy assignment-era evaluation entry point referred to a smaller x0.5 model; the Phase 0 evaluation path now uses the configured x1.0 checkpoint, writes class-wise metrics and a confusion matrix, and records the manifest/checkpoint identities.
4. Training/validation membership is now deterministic and keeps byte-identical training images in the same split. The validator reports 41 exact same-split duplicate groups and capped near-duplicate review candidates; these were retained and documented rather than deleting user data.
5. The fresh run used the available Python 3.10 environment with PyTorch 2.5.1, torchvision 0.20.1, and Pillow 12.1.1 on CPU. CUDA was unavailable to this runtime. This is a successful baseline run, but it does not verify the project's pinned PyTorch 2.6.0 / torchvision 0.21.0 environment.

## Fresh baseline results

- Overall accuracy: **92.44%** over 2,500 images.
- Clean accuracy: **97.28%**; augmented accuracy: **87.60%**.
- Mean CPU inference latency: **9.55 ms/image** in the available environment.
- Per-class support/correct: Bus 500/454, Truck 500/463, Car 500/439, Bike 500/480, None 500/475.
- Confusion matrix is recorded in `reports/baseline.json` and `reports/evaluation.json` using `[Bus, Truck, Car, Bike, None]` order.
- MLflow run: `781f1ac701484fecaffee90434b447f4`.

## Baseline reproduction command and contract

After retrieving the same DVC data version, run `python evaluate_test_set.py --config configs/evaluation.yaml`. This writes the tracked evaluation JSON and MLflow run; `python scripts/record_baseline.py --confirm-canonical` copies the fresh metrics into `reports/baseline.json` only if the checkpoint hash matches the audited canonical checkpoint.

## Decision contract

1. Use the x1.0 `student_model.pth` bytes as the provisional canonical model candidate. Retain the SHA-256 above as its immutable identity.
2. Use the present `test_dataset/` as the provisional target-evaluation candidate, with its current fingerprint recorded above. Provenance and leakage remain unresolved; do not treat this as a versioned dataset release yet.
3. Use the existing deterministic inference preprocessing and class order above as the initial serving contract.
4. Use the fresh 92.44% target result as the provisional baseline gate; the same score in the historical log is still not proof of historical dataset identity. Ratify the threshold with the team.
5. The stable output schema is: class index/name mapping, `overall_accuracy`, per-class support/correct/accuracy, confusion matrix, checkpoint identity, dataset identity, configuration, and elapsed/CPU timing metadata.

## Exit gate status

- [x] Architecture, class mapping, preprocessing, artifact size, and available historical class-wise outputs audited.
- [x] Exact checkpoint bytes tied to the historical 92.44% log entry by SHA-256 equality.
- [x] Fresh baseline inference run in a PyTorch-capable environment (CPU; version caveat above).
- [x] Confusion matrix produced from predictions.
- [x] DVC dataset manifest version recorded and exact cross-split duplicate leakage checked (near-duplicate candidates remain for review).
- [ ] Team ratification of canonical model/data/config/output contract.

