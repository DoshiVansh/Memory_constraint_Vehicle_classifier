#!/usr/bin/env bash
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "DVC is missing. Install the pinned DVC S3 extra with: python -m pip install -r requirements-dvc.txt" >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "Copy .env.example to .env and set a private DVC_SECRET_KEY first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ "${1:-}" == "--start-local-storage" ]]; then
  docker compose --profile dvc-storage up -d rustfs
  echo "Local object storage is starting. Open http://127.0.0.1:9001 and create bucket ${DVC_BUCKET}."
  echo "Then rerun this script without --start-local-storage to configure DVC."
  exit 0
fi

if ! dvc root >/dev/null 2>&1; then
  dvc init
fi

if ! dvc remote list | awk '{print $1}' | grep -qx storage; then
  dvc remote add -d storage "s3://${DVC_BUCKET}/dvc"
else
  dvc remote modify storage url "s3://${DVC_BUCKET}/dvc"
  dvc remote default storage
fi
dvc remote modify --local storage endpointurl "${DVC_ENDPOINT_URL:-http://127.0.0.1:9000}"
dvc remote modify --local storage region us-east-1
dvc remote modify --local storage use_ssl false
dvc remote modify --local storage access_key_id "${DVC_ACCESS_KEY}"
dvc remote modify --local storage secret_access_key "${DVC_SECRET_KEY}"

if [[ "${1:-}" == "--track-local-data" ]]; then
  # DVC creates pointer files and workspace ignore rules. Raw source dumps remain
  # local because their redistribution permissions have not been established.
  dvc add -f vehicle_dataset test_dataset test_MIO_TCD_data
fi
echo "DVC remote configured. Run dvc pull on clients; on the data host run dvc repro && dvc push."
