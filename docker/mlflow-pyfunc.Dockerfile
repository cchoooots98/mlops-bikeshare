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

# Install Python runtime dependencies
# - Pin mlflow to match how you logged the model (3.3.2 per your tags)
# - Gunicorn + Flask provide the web server
# - numpy<2 often avoids ABI surprises with popular libs
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      mlflow==3.3.2 \
      gunicorn==22.0.0 \
      scipy==1.16.1 \
      psutil==7.0.0 \
      flask==3.0.3 \
      pandas==2.3.2 \
      numpy==1.26.4 \
      scikit-learn==1.7.1 \
      cloudpickle==3.1.1 \
      pyarrow==15.0.2 \
      xgboost==3.0.4

# Put our app in the container
WORKDIR /opt/ml/code
COPY docker/app.py /opt/ml/code/app.py
# If your app imports inference.py, copy it too (remove if not used)
COPY docker/inference.py /opt/ml/code/inference.py

# Add the SageMaker-compatible entrypoint wrapper
COPY docker/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# --- Make training schema available at runtime (so serve == train) ---
# Create the Python package path inside the image
RUN mkdir -p /opt/ml/code/src/features

# COPY the schema used during training into the same path inside the image.
# If your repo path is src/features/schema.py, keep the first COPY line.
# If schema.py is at project root, use the second COPY line instead (and remove the first).

# Option A: schema lives at src/features/schema.py in your repo
COPY src/features/schema.py /opt/ml/code/src/features/schema.py

# Option B: schema lives at project root (UNCOMMENT if that's your structure)
# COPY schema.py /opt/ml/code/src/features/schema.py

# Make 'src' importable as a package
RUN bash -lc 'touch /opt/ml/code/src/__init__.py /opt/ml/code/src/features/__init__.py'

# Ensure Python can import from /opt/ml/code
ENV PYTHONPATH=/opt/ml/code

# SageMaker expects the container to listen on 8080
EXPOSE 8080

# SageMaker runs "docker run IMAGE serve" -> this wrapper understands "serve"
ENTRYPOINT ["/usr/local/bin/start.sh"]
CMD ["serve"]