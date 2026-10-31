#!/usr/bin/env bash
# configure_dvc_remote.sh — Configure DVC to use DagsHub as the S3-compatible remote.
# Usage:
#   ./scripts/configure_dvc_remote.sh               # Configure remote only
#   ./scripts/configure_dvc_remote.sh --track-local-data  # Also dvc add local datasets
#
# Prerequisites:
#   1. Copy .env.example → .env and fill in DAGSHUB_USERNAME, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN
#   2. pip install -r requirements-dvc.txt
set -euo pipefail

if ! command -v dvc > /dev/null 2>&1; then
  echo "ERROR: DVC is missing. Install with: pip install -r requirements-dvc.txt" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found. Copy .env.example → .env and fill in your DagsHub credentials." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

# Validate required DagsHub variables are set
: "${DAGSHUB_USERNAME:?Need to set DAGSHUB_USERNAME in .env}"
: "${DAGSHUB_REPO_NAME:?Need to set DAGSHUB_REPO_NAME in .env}"
: "${DAGSHUB_TOKEN:?Need to set DAGSHUB_TOKEN in .env}"

DAGSHUB_S3_ENDPOINT="https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.s3"

if ! dvc root > /dev/null 2>&1; then
  dvc init
fi

# Add or update the 'storage' remote pointing to DagsHub S3
if ! dvc remote list | awk '{print $1}' | grep -qx storage; then
  dvc remote add -d storage "s3://dvc"
else
  dvc remote modify storage url "s3://dvc"
  dvc remote default storage
fi

# Store endpoint and credentials in .dvc/config.local (gitignored — never committed)
# DagsHub S3-compatible remote requires the token for both access_key_id and secret_access_key
dvc remote modify --local storage endpointurl       "${DAGSHUB_S3_ENDPOINT}"
dvc remote modify --local storage access_key_id     "${DAGSHUB_TOKEN}"
dvc remote modify --local storage secret_access_key "${DAGSHUB_TOKEN}"

echo "✅ DVC remote 'storage' configured → ${DAGSHUB_S3_ENDPOINT}"
echo "   Run 'dvc pull' to fetch data or 'dvc push' to upload."

if [[ "${1:-}" == "--track-local-data" ]]; then
  echo "Tracking local datasets with DVC..."
  dvc add -f vehicle_dataset test_dataset test_MIO_TCD_data
  echo "✅ Datasets tracked. Run 'dvc push' to upload to DagsHub."
fi
