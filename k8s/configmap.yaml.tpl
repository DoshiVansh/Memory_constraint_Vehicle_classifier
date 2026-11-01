apiVersion: v1
kind: ConfigMap
metadata:
  name: vehicle-classifier-config
  namespace: vehicle-classifier-staging
  labels:
    app.kubernetes.io/name: vehicle-classifier-api
data:
  MODEL_VERSION: "${MODEL_VERSION}"
  MODEL_SHA256: "${MODEL_SHA256}"
  RELEASE_GIT_SHA: "${RELEASE_GIT_SHA}"
  LOG_LEVEL: "INFO"
  MAX_UPLOAD_BYTES: "8388608"
  MAX_IMAGE_PIXELS: "20000000"
