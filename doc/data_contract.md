# Data Contract — Raw Layer (GBFS + Meteostat)

## Scope
- Ingestion every 5 minutes; UTC timestamps; city partition.
- Landing path: `s3://<bucket>/raw/city=<city>/dt=YYYY-MM-DD-HH-mm/`.

## Required Fields
- **station_status**: station_id, num_bikes_available, num_docks_available, last_reported (epoch sec)
- **station_information**: station_id, name, capacity, lat, lon
- **weather_point_hourly**: time (UTC), temp, prcp, wnd
(Constraints: ranges, types, non-null set…)

## Validation
- Online: Pydantic/Pandera in Lambda; batch refusal on schema error.
- Offline: Great Expectations daily checkpoint on S3 raw (separate job).

## Error Handling
- Reject batch → write to `ingest_errors/`; push CloudWatch metric `IngestFailures`.
- Idempotency key: `city + status.last_updated` → `dt` folder.

## Latency SLO
- ≤ 3 minutes end-to-S3; 99% success. (See Ops SLA)
