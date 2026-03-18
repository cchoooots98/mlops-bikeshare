# Purpose: Streamlit dashboard container.
# - Exposes port 8080.
# - Installs core ML deps from requirements.txt, then dashboard-specific
#   deps from requirements-app.txt (streamlit, plotly, folium, etc.).
# - Copies only app + src code (no tests/mlruns/model_dir).
# - On EC2 Docker Compose: pass STREAMLIT_PG_HOST / STREAMLIT_PG_PORT
#   env vars to override the Postgres address in secrets.toml.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Layer-cache friendly: copy manifests first, install, then copy code
COPY requirements.txt /app/requirements.txt
COPY requirements-app.txt /app/requirements-app.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir -r /app/requirements-app.txt

COPY app/ /app/app/
COPY src/ /app/src/

ENV PORT=8080 \
    AWS_REGION=eu-west-3 \
    CITY=paris \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]
