import argparse
import json
import os

from src.config.naming import endpoint_name
from src.features.postgres_store import PostgresFeatureConfig
from src.monitoring.metrics.publish_psi import (
    DEFAULT_PSI_AGGREGATOR,
    DEFAULT_PSI_QUERY_CHUNK_SIZE,
    PSI_AGGREGATORS,
    compute_psi_result,
    publish_psi_metrics,
)

DEFAULT_TARGETS = ("bikes", "docks")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSI once and publish it for all serving targets.")
    parser.add_argument("--city", default="paris")
    parser.add_argument("--environment", default="production")
    parser.add_argument("--pg-host", required=True)
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-db", required=True)
    parser.add_argument("--pg-user", required=True)
    parser.add_argument("--pg-password", required=True)
    parser.add_argument("--pg-schema", default="analytics")
    parser.add_argument("--feature-table", default="feat_station_snapshot_5min")
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--baseline-days", type=int, default=7)
    parser.add_argument("--aggregator", choices=PSI_AGGREGATORS, default=DEFAULT_PSI_AGGREGATOR)
    parser.add_argument(
        "--max-feature-age-minutes",
        type=int,
        default=int(os.environ.get("PSI_MAX_FEATURE_AGE_MINUTES", "45")),
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=int(os.environ.get("PSI_QUERY_CHUNK_SIZE", str(DEFAULT_PSI_QUERY_CHUNK_SIZE))),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-name", action="append", dest="target_names")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    result = compute_psi_result(
        config=PostgresFeatureConfig(
            pg_host=args.pg_host,
            pg_port=args.pg_port,
            pg_db=args.pg_db,
            pg_user=args.pg_user,
            pg_password=args.pg_password,
            pg_schema=args.pg_schema,
            training_table=args.feature_table,
        ),
        city=args.city,
        lookback_hours=args.lookback_hours,
        baseline_days=args.baseline_days,
        aggregator=args.aggregator,
        max_feature_age_minutes=args.max_feature_age_minutes,
        query_chunk_size=args.query_chunk_size,
    )
    target_names = tuple(args.target_names or DEFAULT_TARGETS)
    publish_targets = [
        {
            "target_name": target_name,
            "endpoint": endpoint_name(target_name=target_name, environment=args.environment),
        }
        for target_name in target_names
    ]
    payload = {
        **result,
        "environment": args.environment,
        "targets": publish_targets,
    }
    if args.dry_run:
        print(json.dumps(payload, default=str))
        return payload

    for target in publish_targets:
        publish_psi_metrics(
            result,
            endpoint=target["endpoint"],
            city=args.city,
            target_name=target["target_name"],
            environment=args.environment,
        )
    print(json.dumps(payload, default=str))
    return payload


if __name__ == "__main__":
    main()
