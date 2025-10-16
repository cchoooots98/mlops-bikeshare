# Purpose: Streamlit dashboard container for AWS App Runner or ECS Fargate.
# Notes:
# - Exposes port 8080 (App Runner default).
# - Installs only runtime deps from requirements.txt.
# - Copies only the app + src code (no tests/mlruns/etc.).
# - Reads env: AWS_REGION, CITY, SM_ENDPOINT, CW_NS.

FROM python:3.11-slim

# Keep image small and deterministic
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system packages if needed by your deps (kept minimal here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first to leverage Docker layer cache
# Make sure requirements.txt includes: streamlit, boto3, pandas, pyarrow, s3fs, plotly, folium, etc.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the minimum runtime code
COPY app/ /app/app/
COPY src/ /app/src/

# Default envs (override in App Runner or docker run -e ...)
ENV PORT=8080 \
    AWS_REGION=ca-central-1 \
    CITY=nyc \
    SM_ENDPOINT=bikeshare-prod \
    CW_NS=Bikeshare/Model \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

# Health: App Runner can check GET /; Streamlit serves 200 on "/"
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]
