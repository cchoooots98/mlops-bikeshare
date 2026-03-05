
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
  observed_at      TIMESTAMP UNIQUE,
  temperature_c    DOUBLE PRECISION,
  dew_point_c      DOUBLE PRECISION,
  humidity_pct     DOUBLE PRECISION,
  precipitation_mm DOUBLE PRECISION,
  snow_mm          DOUBLE PRECISION,
  wind_dir_deg     DOUBLE PRECISION,
  wind_speed_kmh   DOUBLE PRECISION,
  wind_gust_kmh    DOUBLE PRECISION,
  pressure_hpa     DOUBLE PRECISION,
  weather_code     INTEGER
);

CREATE TABLE IF NOT EXISTS fact_station_status (
  station_status_id    BIGSERIAL PRIMARY KEY,
  observed_at          TIMESTAMP NOT NULL,
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