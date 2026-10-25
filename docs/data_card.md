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

After configuring the DVC remote and tracking the three curated directories:

```bash
dvc repro
```

The pipeline creates `data/manifests/data_manifest.json` and `reports/data_validation.json`. Evaluators include the manifest's SHA-256 in their output to identify the exact data contents.
