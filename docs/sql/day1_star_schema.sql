
DROP TABLE IF EXISTS fact_station_status CASCADE;
DROP TABLE IF EXISTS dim_weather CASCADE;
DROP TABLE IF EXISTS dim_time CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_station CASCADE;

CREATE TABLE IF NOT EXISTS dim_station (
  station_version_key TEXT PRIMARY KEY,
  station_key         TEXT NOT NULL,
  city                TEXT NOT NULL,
  station_id          TEXT NOT NULL,
  station_name        TEXT,
  latitude            DOUBLE PRECISION,
  longitude           DOUBLE PRECISION,
  capacity            INTEGER,
  valid_from_utc      TIMESTAMPTZ NOT NULL,
  valid_to_utc        TIMESTAMPTZ,
  is_current          BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS  dim_date (
  date_id         INTEGER PRIMARY KEY,
  date            DATE UNIQUE,
  day_of_week     SMALLINT,
  month           SMALLINT,
  year            SMALLINT,
  is_weekend      BOOLEAN,
  is_holiday      BOOLEAN,
  holiday_name    TEXT
);

CREATE TABLE IF NOT EXISTS dim_time (
  time_id         SMALLINT PRIMARY KEY,
  hour            SMALLINT,
  minute          SMALLINT,
  is_peak_hour    BOOLEAN
);

CREATE TABLE IF NOT EXISTS dim_weather (
  weather_id       BIGSERIAL PRIMARY KEY,
  weather_key                          TEXT UNIQUE,
  city                                 TEXT NOT NULL,
  observed_at                          TIMESTAMPTZ NOT NULL,
  temperature_c                        DOUBLE PRECISION,
  humidity_pct                         DOUBLE PRECISION,
  wind_speed_ms                        DOUBLE PRECISION,
  precipitation_mm                     DOUBLE PRECISION,
  weather_code                         INTEGER,
  weather_main                         TEXT,
  weather_description                  TEXT,
  hourly_forecast_at                   TIMESTAMPTZ,
  hourly_temperature_c                 DOUBLE PRECISION,
  hourly_humidity_pct                  DOUBLE PRECISION,
  hourly_wind_speed_ms                 DOUBLE PRECISION,
  hourly_precipitation_mm              DOUBLE PRECISION,
  hourly_precipitation_probability_pct DOUBLE PRECISION,
  hourly_weather_code                  INTEGER,
  hourly_weather_main                  TEXT,
  source                               TEXT,
  UNIQUE (city, observed_at)
);

CREATE TABLE IF NOT EXISTS fact_station_status (
  fact_station_status_key  TEXT PRIMARY KEY,
  city                     TEXT NOT NULL,
  station_id               TEXT NOT NULL,
  station_key              TEXT NOT NULL,
  station_version_key      TEXT NOT NULL REFERENCES dim_station(station_version_key),
  snapshot_bucket_at_utc   TIMESTAMPTZ NOT NULL,
  snapshot_bucket_at_paris TIMESTAMP NOT NULL,
  date_id                  INTEGER REFERENCES dim_date(date_id),
  time_id                  SMALLINT REFERENCES dim_time(time_id),
  last_reported_at_utc     TIMESTAMPTZ,
  last_reported_at_paris   TIMESTAMP,
  num_bikes_available      INTEGER,
  num_docks_available      INTEGER,
  is_renting               SMALLINT,
  is_returning             SMALLINT,
  UNIQUE (city, snapshot_bucket_at_utc, station_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_station_status_obs
  ON fact_station_status (snapshot_bucket_at_utc);

CREATE INDEX IF NOT EXISTS idx_fact_station_status_station_obs
  ON fact_station_status (city, station_id, snapshot_bucket_at_utc);
