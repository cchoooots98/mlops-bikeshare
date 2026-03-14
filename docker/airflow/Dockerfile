FROM apache/airflow:2.10.3

COPY docker/airflow/requirements-airflow.txt /requirements-airflow.txt

USER airflow

RUN pip install --no-cache-dir -r /requirements-airflow.txt
