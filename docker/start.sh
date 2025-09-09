#!/usr/bin/env bash
# Make the container compatible with SageMaker's "docker run IMAGE serve".
# If called with 'serve', start your Flask app via gunicorn on port 8080.

set -euo pipefail

# SageMaker untars the model here by default
export MODEL_DIR="${MODEL_DIR:-/opt/ml/model}"

# SageMaker expects the HTTP server on port 8080
export PORT="${PORT:-8080}"

case "${1:-serve}" in
  serve)
    # Start the Flask app defined in app.py as 'app'
    exec gunicorn --bind ":${PORT}" --workers "${WORKERS:-1}" --timeout "${TIMEOUT:-600}" app:app
    ;;
  *)
    # Allow arbitrary commands (useful for debugging)
    exec "$@"
    ;;
esac
