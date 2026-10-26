"""Validate Phase 5 release lineage before Jenkins deploys its image digest."""

import argparse
import json
from pathlib import Path
import re


SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
VERSION = re.compile(r"^[1-9][0-9]*$")
IMAGE_NAME = re.compile(r"^ghcr\.io/[a-z0-9][a-z0-9._/-]*$")


def validate(path):
    try:
        release = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read release metadata JSON: {exc}") from exc

    if not isinstance(release, dict):
        raise ValueError("Release metadata root must be a JSON object")
    model = release.get("model")
    image = release.get("image")
    if release.get("schema_version") != 1 or not isinstance(model, dict) or not isinstance(image, dict):
        raise ValueError("Release metadata schema is not supported")
    if model.get("alias") != "champion":
        raise ValueError("Only the approved champion model alias can be deployed")
    version = str(model.get("version", ""))
    if not VERSION.fullmatch(version):
        raise ValueError("Release metadata must contain a positive integer model version")
    if not re.fullmatch(r"[0-9a-f]{64}", str(model.get("artifact_sha256", ""))):
        raise ValueError("Release metadata has an invalid model artifact SHA-256")
    size_bytes = model.get("artifact_size_bytes")
    if type(size_bytes) is not int or not 0 < size_bytes < 5_000_000:
        raise ValueError("Release metadata model artifact violates the strict 5 MB size limit")
    if model.get("architecture") != "shufflenet_v2_x1_0":
        raise ValueError("Release metadata does not identify the approved student architecture")

    image_name = str(image.get("name", ""))
    digest = str(image.get("digest", ""))
    if not IMAGE_NAME.fullmatch(image_name) or not SHA256.fullmatch(digest):
        raise ValueError("Release metadata has an invalid GHCR image name or digest")
    immutable_reference = f"{image_name}@{digest}"
    if image.get("immutable_reference") != immutable_reference:
        raise ValueError("Release metadata immutable image reference does not match its name and digest")
    git_sha = str(release.get("git_sha", ""))
    if not GIT_SHA.fullmatch(git_sha):
        raise ValueError("Release metadata must contain a 40-character Git commit SHA")
    if not release.get("release_tag") or image.get("release_tag") != release.get("release_tag"):
        raise ValueError("Release tag fields are missing or inconsistent")

    return {
        "IMAGE_REF": immutable_reference,
        "MODEL_VERSION": version,
        "MODEL_ALIAS": "champion",
        "MODEL_SHA256": model["artifact_sha256"],
        "RELEASE_GIT_SHA": git_sha,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, help="Write validated Jenkins variables as KEY=VALUE lines")
    args = parser.parse_args()
    try:
        identity = validate(args.metadata)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(identity, sort_keys=True))
    if args.output:
        args.output.write_text("".join(f"{key}={value}\n" for key, value in identity.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
