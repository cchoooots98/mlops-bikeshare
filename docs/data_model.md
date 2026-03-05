# Data Model

## 1. Scope
This document describes the Day2 warehouse data model for GBFS ingestion and the downstream star schema.

## 2. Staging Tables
### stg_station_information
- run_id (text)
- ingested_at (timestamptz)
- source_last_updated (bigint)
- station_id (text)
- name (text)
- lat (double precision)
- lon (double precision)
- capacity (integer)

### stg_station_status
- run_id (text)
- ingested_at (timestamptz)
- source_last_updated (bigint)
- station_id (text)
- num_bikes_available (integer)
- num_docks_available (integer)
- is_renting (smallint)
- is_returning (smallint)
- last_reported_at (timestamptz)

## 3. Star Schema (from Day1)
See: docs/sql/day1_star_schema.sql

## 4. Relationships
- stg_station_information.station_id -> dim_station.station_id (upsert source)
- stg_station_status.station_id -> dim_station.station_id
- fact_station_status references dim_station/dim_date/dim_time/dim_weather

## 5. ER Diagram
![ER Diagram](diagrams/day2_er.png)
