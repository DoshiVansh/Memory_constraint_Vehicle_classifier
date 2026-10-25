# Memory-Constrained Vehicle Classifier

Five-class image classifier using a ShuffleNet V2 x1.0 student distilled from an EfficientNet-B2 teacher. The legacy interface remains available:

```python
from vehicle_classifier import VehicleClassifier

classifier = VehicleClassifier("student_model.pth")
class_index = classifier.predict("vehicle.jpg")  # int in [0, 4]
```

Class order is `0 Bus`, `1 Truck`, `2 Car`, `3 Bike`, `4 None`. Inference runs on CPU, uses RGB 224x224 images with ImageNet normalization, and constructs the architecture without downloading pretrained weights.

## Supported CPU environment

- CPython 3.10, 3.11, or 3.12 (3.13+ is not currently supported by the pinned stack).
- CPU-only PyTorch 2.6.0 and torchvision 0.21.0.
- Pillow 11.1.0, PyYAML 6.0.2, and pytest 8.3.5.
- Linux x86_64 is the reference platform. macOS on Apple silicon/Intel is supported for CPU inference, subject to PyTorch wheel availability. CUDA is not required for inference.

Create a clean environment and install the pinned CPU stack:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.0.1
python -m pip install -r requirements-cpu.txt
python -m pip install Pillow==11.1.0 PyYAML==6.0.2 pytest==8.3.5
python -m pip install -e . --no-deps
```

For GPU training, install the pinned PyTorch/torchvision versions from the official CUDA wheel index matching your driver, then install the remaining packages and this project with `--no-deps` as above. The project configs choose CUDA automatically when that build can access it. PyTorch publishes version-specific CPU and CUDA install commands in its [previous versions guide](https://pytorch.org/get-started/previous-versions/).

## Configuration and commands

Training, evaluation, and data preparation settings live in `configs/`. Configurations set seed `1729` by default; run reports record Python/library versions, platform, Git revision, config, and checkpoint hash. Training data must use the explicit folder names `0_Bus`, `1_Truck`, `2_Car`, `3_Bike`, and `4_None`.

Teacher and student training default to `device: auto`: they use CUDA when the installed PyTorch build can access it, otherwise CPU. The pinned install command above installs the CPU build; GPU training requires installing a CUDA-enabled PyTorch/torchvision build compatible with the system's GPU driver. Inference intentionally remains CPU-only.

```bash
# Install the pinned DVC S3 support. DVC and local RustFS storage are free.
python -m pip install -r requirements-dvc.txt
python -m pip install -r requirements-mlflow.txt

# Create local credentials only if .env does not already exist; .env is ignored
# by Git. Replace DVC_SECRET_KEY with a private value in a newly copied file.
test -f .env || cp .env.example .env
# Start the local DVC object store.
scripts/configure_dvc_remote.sh --start-local-storage

# Open http://127.0.0.1:9001, sign in with .env credentials, create the
# vehicle-classifier-data bucket, then configure DVC and add the existing data:
scripts/configure_dvc_remote.sh --track-local-data

# Build the manifest, validate, and upload to RustFS without training.
dvc repro validate_data
dvc push

# On another team computer, configure its .env with the storage host's
# DVC_ENDPOINT_URL and shared team credentials, configure the remote, then pull.
scripts/configure_dvc_remote.sh
dvc pull

# Train the EfficientNet-B2 teacher (pretrained weights are downloaded for training)
python train_teacher.py --config configs/teacher.yaml

# Train the ShuffleNet student by knowledge distillation
python train_distillation.py --config configs/distillation.yaml

# Evaluate a selected checkpoint and write metrics plus a confusion matrix
python evaluate_test_set.py --config configs/evaluation.yaml

# Optional out-of-distribution stress test
python evaluate_mio_tcd.py --config configs/ood_evaluation.yaml

# Run the interface, deterministic split, and metric checks
pytest
```

## MLflow experiments and model promotion

MLflow is configured for a free, local setup: run metadata is stored in
`mlflow.db` (SQLite), and artifacts are stored under `mlartifacts/`. Both are
ignored by Git. This local store is for this machine and is not automatically
shared with teammates. Set `MLFLOW_TRACKING_URI` to point the scripts and UI at
the same database if you change the default. `MLFLOW_EXPERIMENT_NAME` can select
a different experiment name.

Start the UI from the repository root in one terminal:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Run experiments in another terminal. Each invocation creates a new run; change
the YAML values or config file for the second experiment. The dataset manifest
must first exist (Phase 2 `dvc repro`):

```bash
python train_teacher.py --config configs/teacher.yaml
python train_distillation.py --config configs/distillation.yaml
python evaluate_test_set.py --config configs/evaluation.yaml
python evaluate_mio_tcd.py --config configs/ood_evaluation.yaml
```

Open <http://127.0.0.1:5000> to compare runs. Training runs log seed,
hyperparameters, teacher/student identifiers, teacher checkpoint hash where
applicable, Git SHA/working-tree state, data-manifest or DVC-lock fingerprint,
epoch and final metrics, config, and checkpoints. Evaluation runs additionally
log overall, clean/augmented and per-class metrics, confusion matrix, report,
and exact evaluated checkpoint. If no DVC manifest/lock is available, the run
is explicitly tagged `unversioned-local-data`; that run cannot pass promotion.

After committing the implementation and obtaining a fresh evaluation of the
audited canonical checkpoint on the same data manifest, record the baseline
with explicit confirmation, then use the candidate flow:

```bash
python evaluate_test_set.py --config configs/evaluation.yaml
python scripts/record_baseline.py --evaluation reports/evaluation.json --confirm-canonical
# After training/evaluating a candidate on the same manifest:
python scripts/model_gate.py --evaluation reports/evaluation.json
python scripts/register_candidate.py --evaluation reports/evaluation.json
# Review the candidate and its quality-gate report in MLflow, then explicitly approve:
python scripts/promote_model.py --model-name VehicleClassifier --version 1 --approve
```

The baseline recorder checks that the evaluated checkpoint hash matches the
audited canonical checkpoint and requires a clean committed working tree. The
candidate gate requires its own clean Git revision, MLflow run ID, the same
dataset-manifest checksum as the baseline, matching checkpoint SHA-256,
accuracy at least equal to the fresh canonical baseline, and a checkpoint
strictly below 5,000,000 bytes. The
historical 92.44% in the current baseline audit is not treated as a fresh gate
threshold. Registration creates a candidate version; assigning the `champion`
alias is a separate command requiring `--approve`. The registered version
points to the checkpoint artifact logged by the evaluation run; serving still
uses the existing `VehicleClassifier` loader with the checkpoint file.

The registered artifact is also an MLflow PyFunc model. After promotion it can
be loaded by its alias:

```python
import mlflow
import pandas as pd

model = mlflow.pyfunc.load_model("models:/VehicleClassifier@champion")
predictions = model.predict(pd.DataFrame({"image_path": ["vehicle.jpg"]}))
```

This wrapper calls the same `VehicleClassifier` implementation and returns
integer class indices.

DVC can execute the manifest and validation stages with `dvc repro validate_data`.
The complete ML pipeline (which trains the teacher and student) can be run with
`dvc repro evaluate_target evaluate_ood` when the Phase 2 data and teacher
pretraining weights are available; this may take substantial time.
The MLflow database/artifacts remain local side effects and are not DVC outputs.
The current DVC remote is not yet initialized or populated, and the canonical
baseline has not yet been freshly reproduced, so the promotion flow will remain
blocked until those inputs exist. See [phase status](docs/phase_status.md).

`vehicle_dataset`, `test_dataset`, and `test_MIO_TCD_data` are stored in DVC, not Git. The raw `imagenet_raw/` and `mio_tcd_raw/` directories stay local until their redistribution rights are verified. RustFS stores data on the host disk and has no cloud fee; its single-node mode has no redundancy, so keep a backup. It binds to localhost by default. Sharing across computers requires binding to a trusted LAN, setting each client's `DVC_ENDPOINT_URL` to the host address, and keeping RustFS running. Do not expose this development endpoint to the public internet. Keep `.obsidian/` and `.env` private; both are ignored by Git.

The legacy `train.py` entry point delegates to the configured distillation workflow. The historical student checkpoint is small enough for the 5,000,000-byte artifact cap; newly trained artifacts are checked against the same cap and accompanied by a JSON metadata file.

## Code layout

- `vehicle_classifier_core/model.py`: model architecture and class map.
- `vehicle_classifier_core/preprocessing.py`: shared training and inference transforms.
- `vehicle_classifier_core/data.py`: class-map validation and seeded train/validation split.
- `vehicle_classifier_core/predictor.py`: CPU inference; exported through `vehicle_classifier.py` for compatibility.
- `vehicle_classifier_core/evaluation.py`: classwise metrics and confusion matrix.
- `vehicle_classifier_core/packaging.py`: checkpoint manifest, checksum, and size.
- `vehicle_classifier_core/reproducibility.py`: random seeds, JSON logs, run metadata.
- `vehicle_classifier_core/tracking.py`: MLflow run metadata and artifact logging.
- `vehicle_classifier_core/model_gate.py`: fresh-baseline and strict artifact-size gate.
- `docs/baseline_audit.md`: Phase 0 audit and current baseline status.
- `docs/phase_status.md`: implementation progress versus pending exit-gate evidence.
