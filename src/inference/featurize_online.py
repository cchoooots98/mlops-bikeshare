from typing import Sequence

import pandas as pd

from src.config import load_runtime_settings
from src.features.postgres_store import PostgresFeatureConfig, create_pg_engine, load_latest_serving_features
from src.features.schema import ENTITY_COLUMNS, FEATURE_COLUMNS, REQUIRED_BASE


def build_online_features(
    city: str | None = None,
    *,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    settings = load_runtime_settings()
    target_city = city or settings.city
    selected_feature_columns = list(feature_columns or FEATURE_COLUMNS)
    pg_config = PostgresFeatureConfig(
        pg_host=settings.pg_host,
        pg_port=settings.pg_port,
        pg_db=settings.pg_db,
        pg_user=settings.pg_user,
        pg_password=settings.pg_password,
        pg_schema=settings.pg_schema,
        training_table=settings.training_feature_table,
        online_table=settings.online_feature_table,
    )
    engine = create_pg_engine(pg_config)
    features = load_latest_serving_features(
        engine,
        pg_config,
        target_city,
        select_columns=[*ENTITY_COLUMNS, *REQUIRED_BASE, *selected_feature_columns],
        max_dt_skew_minutes=settings.serving_feature_max_dt_skew_minutes,
    )
    return features[[*ENTITY_COLUMNS, *selected_feature_columns]].copy()


if __name__ == "__main__":
    settings = load_runtime_settings()
    out = build_online_features(settings.city)
    print(out.head(5).to_string(index=False))
