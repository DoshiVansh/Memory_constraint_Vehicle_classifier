from vehicle_classifier_core.data_versioning import validate_manifest


def test_validator_flags_exact_hash_crossing_splits():
    manifest = {
        "datasets": {"training": {}, "target": {}, "ood": {}},
        "records": [
            {
                "dataset": "training",
                "split": "train",
                "class_name": "Bus",
                "path": "0_Bus/a.jpg",
                "sha256": "same",
                "dhash64": "0000000000000000",
                "decode_error": None,
                "width": 10,
                "height": 10,
                "size_bytes": 1,
            },
            {
                "dataset": "target",
                "split": "test_clean",
                "class_name": "Bus",
                "path": "Bus/clean_a.jpg",
                "sha256": "same",
                "dhash64": "0000000000000000",
                "decode_error": None,
                "width": 10,
                "height": 10,
                "size_bytes": 1,
            },
        ],
    }

    report = validate_manifest(manifest)

    assert report["status"] == "failed"
    assert any(error["type"] == "duplicate_across_splits" for error in report["errors"])
