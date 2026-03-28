import os

from airflow.hooks.base import BaseHook
from airflow.models import Variable


def get_airflow_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def get_dw_connection():
    conn_id = get_airflow_setting("DW_CONN_ID", "DW_CONN_ID", "velib_dw")
    return BaseHook.get_connection(conn_id)


def normalize_postgres_conn_uri(uri: str) -> str:
    if uri.startswith("postgres://"):
        return uri.replace("postgres://", "postgresql+psycopg2://", 1)
    if uri.startswith("postgresql://"):
        return uri.replace("postgresql://", "postgresql+psycopg2://", 1)
    return uri


def get_dw_conn_uri() -> str:
    return normalize_postgres_conn_uri(get_dw_connection().get_uri())
