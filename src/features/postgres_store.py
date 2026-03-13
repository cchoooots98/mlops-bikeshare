from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

from src.model_target import target_spec_from_predict_bikes
from src.features.schema import (
    ENTITY_COLUMNS,
    LABEL_COLUMNS,
    ONLINE_REQUIRED_COLUMNS,
    REQUIRED_BASE,
    TRAINING_REQUIRED_COLUMNS,
    validate_feature_df,
)

IDENTIFIER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


@dataclass(frozen=True)
class PostgresFeatureConfig:
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str
    pg_schema: str = "analytics"
    training_table: str = "feat_station_snapshot_5min"
    online_table: str = "feat_station_snapshot_latest"


def validate_identifier(identifier: str) -> str:
    if not identifier:
        raise ValueError("SQL identifier must be non-empty")
    if not (identifier[0].isalpha() or identifier[0] == "_"):
        raise ValueError(f"invalid SQL identifier: {identifier}")
    if any(char not in IDENTIFIER_CHARS for char in identifier):
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return identifier


def create_pg_engine(config: PostgresFeatureConfig) -> Engine:
    url = URL.create(
        "postgresql+psycopg2",
        username=config.pg_user,
        password=config.pg_password,
        host=config.pg_host,
        port=config.pg_port,
        database=config.pg_db,
    )
    return create_engine(url, pool_pre_ping=True)


def build_feature_select_query(
    schema_name: str,
    table_name: str,
    select_columns: Sequence[str],
    where_sql: str,
    order_sql: str,
) -> str:
    schema = validate_identifier(schema_name)
    table = validate_identifier(table_name)
    select_list = ", ".join([f'"{column}" AS "{column}"' for column in select_columns])
    return f'SELECT {select_list} FROM "{schema}"."{table}" WHERE {where_sql} {order_sql}'


def _selected_feature_columns(columns: Sequence[str]) -> list[str]:
    excluded = set(ENTITY_COLUMNS + REQUIRED_BASE + LABEL_COLUMNS)
    return [column for column in columns if column not in excluded]


def list_unique_dt_postgres(engine: Engine, config: PostgresFeatureConfig, city: str, start_dt: str, end_dt: str) -> list[str]:
    sql = (
        f'SELECT DISTINCT dt FROM "{validate_identifier(config.pg_schema)}".'
        f'"{validate_identifier(config.training_table)}" '
        "WHERE city = :city AND dt >= :start_dt AND dt <= :end_dt "
        "ORDER BY dt"
    )
    params = {"city": city, "start_dt": start_dt, "end_dt": end_dt}
    with engine.connect() as connection:
        return pd.read_sql_query(text(sql), connection, params=params)["dt"].tolist()


def load_training_slice(
    engine: Engine,
    config: PostgresFeatureConfig,
    city: str,
    start_dt: str,
    end_dt: str,
    select_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns = list(select_columns or TRAINING_REQUIRED_COLUMNS)
    sql = build_feature_select_query(
        config.pg_schema,
        config.training_table,
        columns,
        "city = :city AND dt >= :start_dt AND dt <= :end_dt",
        "ORDER BY dt, station_id",
    )
    params = {"city": city, "start_dt": start_dt, "end_dt": end_dt}
    with engine.connect() as connection:
        df = pd.read_sql_query(text(sql), connection, params=params)
    validate_feature_df(df, require_labels=True, feature_columns=_selected_feature_columns(columns))
    return df


def load_latest_serving_features(
    engine: Engine,
    config: PostgresFeatureConfig,
    city: str,
    select_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    columns = list(select_columns or ONLINE_REQUIRED_COLUMNS)
    sql = build_feature_select_query(
        config.pg_schema,
        config.online_table,
        columns,
        "city = :city",
        "ORDER BY dt, station_id",
    )
    with engine.connect() as connection:
        df = pd.read_sql_query(text(sql), connection, params={"city": city})
    validate_feature_df(df, require_labels=False, feature_columns=_selected_feature_columns(columns))
    label_columns_present = sorted(set(df.columns).intersection(LABEL_COLUMNS))
    if label_columns_present:
        raise ValueError(f"serving feature table must not expose label columns: {label_columns_present}")
    return df


def load_training_actuals_for_dt(
    engine: Engine,
    config: PostgresFeatureConfig,
    city: str,
    dt: str,
    predict_bikes: bool = True,
) -> pd.DataFrame:
    target_spec = target_spec_from_predict_bikes(predict_bikes)
    sql = build_feature_select_query(
        config.pg_schema,
        config.training_table,
        ["station_id", target_spec.paired_target_column, target_spec.label_column],
        "city = :city AND dt = :dt",
        "ORDER BY station_id",
    )
    with engine.connect() as connection:
        return pd.read_sql_query(text(sql), connection, params={"city": city, "dt": dt})
