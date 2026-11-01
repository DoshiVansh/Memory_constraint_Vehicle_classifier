# Model card: VehicleClassifier v2 (champion)

Updated 2026-10-26. This card describes the locally approved package and the
evaluation evidence in this repository. It is not a claim of production
readiness.

## Model identity

| Field | Value |
|---|---|
| Registered name / version / alias | `VehicleClassifier` / `2` / `champion` |
| Architecture | ShuffleNet V2 x1.0 student, distilled from EfficientNet-B2 |
| Checkpoint | `student_model.pth`, 2,665,366 bytes (2.54 MiB) |
| SHA-256 | `c80eaf7a45e87e604b2e41af992094425834e46d747315a6fb66a65f7c18e1d4` |
| Input | RGB image resized to 224×224; ImageNet mean/std normalization |
| Output | Five-class index: 0 Bus, 1 Truck, 2 Car, 3 Bike, 4 None |
| Runtime | CPU inference; API image contains weights and needs no runtime weight download |

## Intended and out-of-scope use

The package is a student project demonstrating reproducible model packaging,
evaluation, API serving, and operational monitoring. It classifies a single
vehicle-like image into the five project labels. It is not validated for
traffic enforcement, safety-critical decisions, autonomous driving, identity
or license-plate recognition, or broad real-world deployment. `None` is the
dataset's fifth class, not a calibrated open-set rejection mechanism.

## Evaluation

The fresh CPU evaluation used seed 1729, the checked-in checkpoint, and target
manifest SHA `3c40f2a7c554220c35c98db4b299a29b3f9c6391f2d56a31695796bc8ccfd35a`.
The target MLflow run is `8099c25e7b824646828fc90baa1e3b01`; the OOD run is
`7eb922a3882349d48121f4bf91024cfb`.
The fixed target set has 2,500 samples (500/class), including 1,250 clean and
1,250 fixed augmented samples. Overall accuracy was **92.40%**; clean accuracy
**97.36%**; augmented accuracy **87.44%**. Per-class target accuracy:

| Class | Correct / 500 | Accuracy |
|---|---:|---:|
| Bus | 449 | 89.8% |
| Truck | 464 | 92.8% |
| Car | 442 | 88.4% |
| Bike | 479 | 95.8% |
| None | 476 | 95.2% |

The mapped MIO-TCD stress set has 2,500 samples and **43.24%** accuracy:
Bus 22.6%, Truck 23.4%, Car 82.2%, Bike 22.0%, None 66.0%. This is a strong
warning that the model does not generalize to this external distribution. The
mapping merges MIO-TCD pedestrian/background categories into `None`, which is
not semantically equivalent to the training class. See the complete confusion
matrices and subgroup metrics in `reports/evaluation.json` and
`reports/ood_evaluation.json`.

Target evaluation mean CPU time was about 11.1 ms/image in the isolated
Python 3.10 container for this run. API load results in
`reports/phase7_local_load.json` are synthetic, local kind measurements, not an
SLO or production capacity estimate.

## Data and known limitations

Data counts, curation, validation and provenance caveats are in
[`data_card.md`](data_card.md). The source archives, license acceptance,
original sample IDs and exact source-to-curated mapping remain unverified.
Evaluation is based on fixed local datasets and may contain dataset-specific
biases. The OOD result is not an estimate of broad real-world accuracy.

No calibrated confidence threshold, fairness audit, temporal validation,
geographic validation, or labeled production feedback pipeline is available.
Prometheus class counts are an unlabeled distribution signal and cannot
measure accuracy. Retrain/re-evaluate with verified provenance and relevant
deployment-domain labels before using this model beyond a demonstration.
