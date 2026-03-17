# docker/mlflow-pyfunc.Dockerfile
# Purpose: Minimal, SageMaker-compatible inference image using Flask + Gunicorn + MLflow PyFunc.

FROM python:3.11-slim

# Make Python logs unbuffered and avoid .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Optional system packages (add only what you need)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies.
# Keep these aligned with the local training environment so the saved MLflow
# package can be deserialized and executed without version drift warnings.
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      mlflow==3.10.1 \
      gunicorn==22.0.0 \
      scipy==1.17.1 \
      psutil==7.0.0 \
      flask==3.0.3 \
      pandas==2.3.3 \
      numpy==2.4.3 \
      scikit-learn==1.8.0 \
      cloudpickle==3.1.2 \
      pyarrow==22.0.0 \
      xgboost==3.2.0

# Put our app in the container
WORKDIR /opt/ml/code
COPY docker/app.py /opt/ml/code/app.py
# If your app imports inference.py, copy it too (remove if not used)
COPY docker/inference.py /opt/ml/code/inference.py

# Add the SageMaker-compatible entrypoint wrapper
COPY docker/start.sh /usr/local/bin/start.sh
RUN sed -i 's/\r$//' /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# --- Make runtime Python modules available inside the image ---
# Create the Python package path inside the image
RUN mkdir -p /opt/ml/code/src/features

COPY src/features/schema.py /opt/ml/code/src/features/schema.py
COPY src/mlflow_pyfunc_model.py /opt/ml/code/src/mlflow_pyfunc_model.py

# Make 'src' importable as a package
RUN bash -lc 'touch /opt/ml/code/src/__init__.py /opt/ml/code/src/features/__init__.py'

# Ensure Python can import from /opt/ml/code
ENV PYTHONPATH=/opt/ml/code

# SageMaker expects the container to listen on 8080
EXPOSE 8080

# SageMaker runs "docker run IMAGE serve" -> this wrapper understands "serve"
ENTRYPOINT ["/usr/local/bin/start.sh"]
CMD ["serve"]
