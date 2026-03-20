FROM apache/airflow:2.10.3

ARG AIRFLOW_VERSION=2.10.3
ARG PYTHON_VERSION=3.12

COPY docker/airflow/requirements-airflow.txt /requirements-airflow.txt

USER airflow

RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
    -r /requirements-airflow.txt
