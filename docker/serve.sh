#!/usr/bin/env bash
set -euo pipefail
# Serve MLflow pyfunc model from /opt/ml/model.
# We use --env-manager local, so all runtime deps must be preinstalled in the image.
exec mlflow models serve \
  -m "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --env-manager local
