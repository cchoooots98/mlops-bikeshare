FROM apache/airflow:2.10.3

USER root

COPY requirements.txt /requirements-project.txt
COPY docker/airflow/requirements-project-runtime.txt /requirements-project-runtime.txt

RUN python -m venv /opt/project-venv && \
    /opt/project-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/project-venv/bin/pip install --no-cache-dir -r /requirements-project.txt -r /requirements-project-runtime.txt && \
    chown -R airflow:0 /opt/project-venv

USER airflow

ENV PROJECT_RUNTIME_PYTHON=/opt/project-venv/bin/python
ENV PROJECT_RUNTIME_DBT=/opt/project-venv/bin/dbt
