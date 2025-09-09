# inference.py
# Minimal SageMaker-compatible handler for an MLflow PyFunc model.

import json
import mlflow.pyfunc
import pandas as pd

def model_fn(model_dir):
    # Load the MLflow PyFunc model that SageMaker untars to /opt/ml/model
    model = mlflow.pyfunc.load_model(model_dir)
    return model

def input_fn(request_body, request_content_type):
    # Accept JSON; prefer MLflow pandas split orientation
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        # Expect {"inputs":{"dataframe_split":{"columns":[...],"data":[...]}}}
        if "inputs" in payload and "dataframe_split" in payload["inputs"]:
            df = pd.DataFrame(**payload["inputs"]["dataframe_split"])
            return df
        # Fallback if user sends {"data":[...]}
        if "data" in payload:
            return pd.DataFrame(payload["data"])
        raise ValueError("Unsupported JSON structure. Expect 'inputs.dataframe_split' or 'data'.")
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # Run inference using the MLflow PyFunc model
    preds = model.predict(input_data)
    return preds

def output_fn(prediction, accept):
    # Return JSON
    if accept == "application/json":
        if hasattr(prediction, "tolist"):
            out = {"predictions": prediction.tolist()}
        else:
            out = {"predictions": prediction}
        return json.dumps(out), accept
    raise ValueError(f"Unsupported accept type: {accept}")
