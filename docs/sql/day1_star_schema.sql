
DROP TABLE IF EXISTS fact_station_status CASCADE;
DROP TABLE IF EXISTS dim_weather CASCADE;
DROP TABLE IF EXISTS dim_time CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_station CASCADE;

CREATE TABLE IF NOT EXISTS dim_station (
  station_id      TEXT PRIMARY KEY,
  name            TEXT,
  latitude        DOUBLE PRECISION,
  longitude       DOUBLE PRECISION,
  capacity        INTEGER
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
  current_precipitation_mm             DOUBLE PRECISION,
  next_hour_precipitation_mm           DOUBLE PRECISION,
  next_hour_precipitation_probability_pct DOUBLE PRECISION,
  rain_next_hour_flag                  BOOLEAN,
  next_hour_valid_at                   TIMESTAMPTZ,
  weather_code                         INTEGER,
  weather_main                         TEXT,
  weather_description                  TEXT,
  source                               TEXT,
  UNIQUE (city, observed_at)
);

CREATE TABLE IF NOT EXISTS fact_station_status (
  station_status_id    BIGSERIAL PRIMARY KEY,
  observed_at          TIMESTAMPTZ NOT NULL,
  station_id           TEXT NOT NULL REFERENCES dim_station(station_id),
  date_id              INTEGER REFERENCES dim_date(date_id),
  time_id              SMALLINT REFERENCES dim_time(time_id),
  weather_id           BIGINT REFERENCES dim_weather(weather_id),
  num_bikes_available  INTEGER,
  num_docks_available  INTEGER,
  is_renting           SMALLINT,
  is_returning         SMALLINT,
  bike_shortage_30m    SMALLINT,
  UNIQUE (observed_at, station_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_station_status_obs
  ON fact_station_status (observed_at);

CREATE INDEX IF NOT EXISTS idx_fact_station_status_station_obs
  ON fact_station_status (station_id, observed_at);
