# Data card (Phase 2 draft)

## Versioned datasets

The DVC pipeline versions three local curated datasets as exact inputs and writes a manifest with paths relative to each dataset, class index, split, source note, dimensions, file size, SHA-256, and a 64-bit difference hash for duplicate review.

| DVC dataset | Role | Current local count | Current local size |
| --- | --- | ---: | ---: |
| `vehicle_dataset` | Training pool; deterministic train/validation membership is recorded in the manifest | 19,690 | about 2.0 GB |
| `test_dataset` | Target evaluation; clean and fixed corrupted examples | 2,500 | about 28 MB |
| `test_MIO_TCD_data` | MIO-TCD mapped OOD stress set | 2,500 | about 21 MB |

Current training counts are not uniform: Bus 4,000; Truck 3,811; Car 4,000; Bike 4,000; None 3,879. This differs from `configs/data.yaml`'s *future curation cap* of 3,000 per class. The existing directories contain extra files from earlier preparation, so the cap must not be interpreted as a description of this current dataset version.

## Provenance and limitations

- The repository contains ImageNet-synset-like source files in `imagenet_raw/` and MIO-TCD-like files in `mio_tcd_raw/`, but the exact download archive, license acceptance, original sample IDs, and source-to-curated mapping are not documented. The manifest records this provenance as unverified rather than asserting it.
- Raw source dumps are intentionally excluded from DVC push for now. Versioning and sharing them would require confirming redistribution rights. The DVC version will contain the curated training, target-test, and mapped OOD files needed to reproduce current runs.
- The OOD set maps MIO-TCD `pedestrian` and `background` categories into the project's `None` class. This ontology mismatch remains a limitation.
- The manifest detects unreadable images, class omissions, exact duplicates, cross-split exact leakage, and possible near duplicates using dHash. Near-duplicate matches are review warnings, not automatic proof of leakage.
- The existing test and OOD sets are fixed files, not regenerated during `dvc repro`; this keeps corruption examples stable across model runs.

## Rebuild and validate

After installing the project dependencies, ensure the datasets have been
retrieved with `dvc pull`, then rebuild only the Phase 2 data manifest/report:

```bash
dvc repro validate_data
```

The full pipeline command (`dvc repro evaluate_target evaluate_ood`) also runs
teacher and student training and can take substantial time. The current data
manifest contains 24,690 image records and is built from the files materialized
by DVC.

Latest local validation: `review_warnings`, zero errors. Exact duplicates were
kept within the same split group so they do not cross train/validation/test
boundaries. The report includes 41 exact same-split duplicate groups and 209
near-duplicate examples for review (the near-duplicate scan is capped). These
are visible in `reports/data_validation.json`; they have not been silently
removed from the user's dataset. Provenance and the OOD `None` ontology
limitation above remain unresolved.

Evaluators include the manifest's SHA-256 in their output to identify the exact
data contents. The DVC data pointers and manifest cache have been pushed to
the local RustFS remote. A temporary clean checkout successfully retrieved all
three datasets and rebuilt the identical manifest SHA-256 and validation report.
