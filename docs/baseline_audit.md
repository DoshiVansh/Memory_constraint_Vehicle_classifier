# Phase 0 baseline audit

**Audit date:** 2026-10-25  
**Repository revision audited:** `732d1eeaadcd908e311f081d70a53bc606b5d042`  
**Status:** Audit facts are recorded; a fresh inference run is pending a PyTorch-capable environment.

## Selected baseline candidate

- **Architecture:** torchvision ShuffleNet V2 x1.0, five output logits (`model.fc`). This agrees with `vehicle_classifier.py` and the distillation script's intended student.
- **Checkpoint:** `student_model.pth`, 2,665,366 bytes (2.54 MiB; 2.67 decimal MB), SHA-256 `c80eaf7a45e87e604b2e41af992094425834e46d747315a6fb66a65f7c18e1d4`.
- **Checkpoint identity:** The untracked `shuffleNet_v2_model.pth` has the same SHA-256, so it is byte-identical to the selected checkpoint. The historical `model_test_log.txt` reports 92.44% on 2,500 images for that filename. This ties the historical score to these exact bytes, but it does not establish that the current dataset contents are identical to those used for that historical run.
- **Dataset candidate:** `test_dataset/`, 2,500 images: 500 each in `Bus`, `Truck`, `Car`, `Bike`, and `None`. The current test tree aggregate (sorted per-file SHA-256 list hashed as bytes) is `34ce9044e7b87f67c45208254470486e59680518e0f2bfd0c787876e1d20c4ac`.
- **Class contract:** `{0: Bus, 1: Truck, 2: Car, 3: Bike, 4: None}`. Training folders are prefixed `0_Bus`, `1_Truck`, `2_Car`, `3_Bike`, `4_None`; this makes `ImageFolder`'s alphabetical order agree with the contract. Evaluation folders are unprefixed and the legacy evaluator explicitly maps them to the same indices.
- **Preprocessing in serving:** RGB conversion; resize directly to 224x224; tensor conversion; ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Distillation validation uses the same resize and normalization. Training adds random augmentation before normalization.
- **Serialized size gate:** The selected artifact is below the strict 5,000,000-byte limit. This is file size only, not peak process/container RAM.

## Findings and inconsistencies

1. The serving model factory previously requested `ShuffleNet_V2_X1_0_Weights.DEFAULT` even though the full checkpoint is loaded afterward. That could fetch an unnecessary pretrained file and prevented reliable offline inference. It now constructs with `weights=None`; loading the full trained state remains required when a checkpoint is supplied.
2. Several checkpoints and historical logs coexist. The 92.44% log entry is for `shuffleNet_v2_model.pth`, which is byte-identical to `student_model.pth`; the smaller x0.5 model has a separate historical score of 91.28%. Do not mix them.
3. The legacy evaluation script currently points at `shuffleNet_v2_0.5_model.pth`, while the report's canonical intended architecture and 92.44% result refer to ShuffleNet V2 x1.0. The script also writes logs while running and reports no confusion matrix.
4. The training pipeline uses a random, unseeded train/validation split and random augmentations. The existing test set is stored as files, but no tracked manifest or data version identifies its provenance. The current tree hash above fingerprints present files; it does not prove provenance or train/test disjointness.
5. No fresh run, confusion matrix, per-class predictions, or CPU latency measurement was produced in this environment: `python3` is available, but PyTorch is not installed (`ModuleNotFoundError: No module named 'torch'`). The historical log gives per-class accuracy but not the confusion matrix.

## Baseline reproduction command and contract

After installing the project's compatible PyTorch, torchvision, and Pillow dependencies, the baseline runner should evaluate exactly `student_model.pth` against `test_dataset/` with the explicit folder-to-index mapping above and emit `reports/baseline.json` plus a 5x5 confusion matrix in true-label/predicted-label order `[Bus, Truck, Car, Bike, None]`. The runner must record checkpoint hash/size, data fingerprint, code revision, preprocessing, sample counts, overall and per-class metrics, confusion matrix, and CPU timing. The current JSON records historical values separately from fresh-run status.

## Decision contract

1. Use the x1.0 `student_model.pth` bytes as the provisional canonical model candidate. Retain the SHA-256 above as its immutable identity.
2. Use the present `test_dataset/` as the provisional target-evaluation candidate, with its current fingerprint recorded above. Provenance and leakage remain unresolved; do not treat this as a versioned dataset release yet.
3. Use the existing deterministic inference preprocessing and class order above as the initial serving contract.
4. Use the legacy reported 92.44% only as a historical reference, not a verified release threshold. Agree on thresholds after a fresh, traceable run.
5. The stable output schema is: class index/name mapping, `overall_accuracy`, per-class support/correct/accuracy, confusion matrix, checkpoint identity, dataset identity, configuration, and elapsed/CPU timing metadata.

## Exit gate status

- [x] Architecture, class mapping, preprocessing, artifact size, and available historical class-wise outputs audited.
- [x] Exact checkpoint bytes tied to the historical 92.44% log entry by SHA-256 equality.
- [ ] Fresh baseline inference run in a PyTorch-capable environment.
- [ ] Confusion matrix produced from predictions.
- [ ] Dataset provenance/version and split leakage established.
- [ ] Team ratification of canonical model/data/config/output contract.

