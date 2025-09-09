# docker/app.py
# Minimal Flask + MLflow PyFunc inference server for SageMaker BYOC.
# - /ping: always returns 200 (so the container is considered healthy even before model download)
# - /invocations: loads MLflow model from /opt/ml/model on demand and runs predict
# - Input format: JSON with MLflow pandas 'split' orientation or a simple {"data":[...]} fallback.

import os
import json
import traceback
from typing import Optional

import mlflow.pyfunc
import pandas as pd
from flask import Flask, request, jsonify, Response
try:
    from src.features.schema import FEATURE_COLUMNS  # training feature list (25 columns)
except ImportError:
    from schema import FEATURE_COLUMNS               # fallback if schema.py is at project root  # exact 25 features used for training
EXPECTED_COLUMNS = FEATURE_COLUMNS  # keep a single source of truth


app = Flask(__name__)

# SageMaker downloads and untars model.tar.gz into this folder at container start.
MODEL_DIR = "/opt/ml/model"

# Cache the loaded model in memory after first successful load.
_cached_model: Optional[mlflow.pyfunc.PyFuncModel] = None


def try_load_model() -> Optional[mlflow.pyfunc.PyFuncModel]:
    """
    Try to load an MLflow PyFunc model from MODEL_DIR.
    Return the cached model if already loaded; return None if not present yet.
    """
    global _cached_model

    # If we already loaded it, reuse it.
    if _cached_model is not None:
        return _cached_model

    # Check for the MLflow marker file; if missing, model is not ready yet.
    mlmodel_path = os.path.join(MODEL_DIR, "MLmodel")
    if not os.path.exists(mlmodel_path):
        return None

    # Load the model (can raise if content is not a valid MLflow pyfunc model).
    _cached_model = mlflow.pyfunc.load_model(MODEL_DIR)
    return _cached_model


@app.get("/ping")
def ping() -> Response:
    """
    Health endpoint.
    Return 200 immediately so SageMaker considers the container healthy even if the model is still downloading.
    """
    return Response("pong", status=200)


@app.post("/invocations")
def invocations():
    """
    Inference endpoint. Expects one of:
      {
        "inputs": {
          "dataframe_split": {
            "columns": [...],
            "data": [[...], ...]
          }
        }
      }
    or a simple fallback:
      {"data": [[...], ...]}
    """
    try:
        model = try_load_model()
        if model is None:
            # The model hasn't been downloaded/unpacked yet (common during cold start).
            return Response("model not ready", status=503)

        # Content-Type must be JSON for this simple handler.
        content_type = request.headers.get("Content-Type", "application/json")
        if content_type != "application/json":
            return Response(f"Unsupported Content-Type: {content_type}", status=415)

        payload = request.get_json(force=True)

        # Preferred MLflow split orientation
        if "inputs" in payload and "dataframe_split" in payload["inputs"]:
            df = pd.DataFrame(**payload["inputs"]["dataframe_split"])
        elif "data" in payload:
            # Fallback: simple 2D array -> DataFrame
            df = pd.DataFrame(payload["data"])
        else:
            return Response(
                "Unsupported JSON structure. Expect 'inputs.dataframe_split' or 'data'.",
                status=400,
            )
        
        # 1) Validate required columns
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            return jsonify({"error":"missing_required_columns", "missing":missing,
                            "expected":EXPECTED_COLUMNS}), 400

        # 2) Reorder to exact training order, drop extras
        df = df.reindex(columns=EXPECTED_COLUMNS)

        # 3) Coerce any string/object to numeric, invalid parses become NaN
        
        df = df.apply(lambda s: pd.to_numeric(s, errors="coerce") if s.dtype == "object" else s)

        # 4) Optionally fill NaN for a robust smoke test (tune for your use case)
        df = df.fillna(0)
            

        # Run prediction
        preds = model.predict(df)
        out = {"predictions": preds.tolist() if hasattr(preds, "tolist") else preds}
        return jsonify(out)

    except Exception as e:
        # Log full traceback for CloudWatch debugging, return generic 500 to client.
        app.logger.error("Invocation failed: %s\n%s", e, traceback.format_exc())
        return Response("error during inference", status=500)
