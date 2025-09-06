# Image: minimal Python 3.11 base
FROM python:3.11-slim

# Install basic certs and curl only (keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Install MLflow (close to your local) + gunicorn web server
# Also install common model deps so pyfunc can load your model without conda:
# - pandas/numpy/cloudpickle are used by MLflow and many flavors
# - scikit-learn (if your pipeline uses it)
# - xgboost (you trained with XGBoost in Step 4)
RUN pip install --no-cache-dir \
    mlflow==2.14.1 \
    gunicorn \
    pandas==2.2.2 \
    numpy \
    cloudpickle \
    scikit-learn \
    xgboost==2.0.3

# SageMaker expects the server on port 8080 and will call /ping and /invocations
EXPOSE 8080

# Where SageMaker untars model.tar.gz (we packed MLmodel at archive root already)
ENV MLFLOW_MODEL_PATH="/opt/ml/model"

# Start the MLflow pyfunc server on 0.0.0.0:8080 (SageMaker health checks depend on this)
# --no-conda/--env-manager local tells MLflow to use the libs baked into THIS image
CMD ["bash","-lc","mlflow models serve -m \"$MLFLOW_MODEL_PATH\" --no-conda --env-manager local --host 0.0.0.0 --port 8080"]
