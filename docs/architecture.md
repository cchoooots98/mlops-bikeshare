# System Architecture

## Overview
The platform is organized around three operating layers:
- data production: Python ingestion plus dbt models in Postgres
- model production: offline training, MLflow logging, and local model packages
- model activation: deployment-state files, staging/prod promotion, and rollback

The formal architecture is hybrid:
- local development for coding, tests, and offline model iteration
- EC2 for always-on data engineering and dashboard hosting
- AWS for serving, alerting, promotion, and rollback

## Component Inventory

| Component | Technology | Role |
|---|---|---|
| Data ingestion | Python 3.11 (`src/ingest/`) | API fetch → raw S3 + staging tables |
| Data warehouse | PostgreSQL 15 | `public.stg_*` staging, `analytics.*` curated |
| Transformation | dbt 1.10 | staging → intermediate → dims/facts → features |
| Orchestration | Apache Airflow 2.10.3 + CeleryExecutor | Scheduled DAGs (EC2-hosted) |
| Message broker | Redis | Celery task queue |
| Experiment tracking | MLflow 3.10.1 | Parameters, metrics, model registry |
| ML training | XGBoost, LightGBM, scikit-learn | Offline training (`src/training/train.py`) |
| Model packaging | `src/model_package/` | Package manifests + deployment state JSON |
| Serving | AWS SageMaker + Lambda | Staging/prod endpoints per target |
| Object storage | AWS S3 | Raw, predictions, quality, PSI, packages |
| Monitoring | CloudWatch + SNS | Custom metrics, alarms, notifications |
| Dashboard | Streamlit (`app/dashboard/`) | Station risk map, model health, freshness |
| Container registry | AWS ECR | Inference container images |
| Infrastructure | Terraform 1.13.3 | `infra/terraform/bootstrap/` + `live/` |
| CI/CD | GitHub Actions | Tests, linting, Terraform validation |

## Architecture Diagrams

### Diagram 1 — System Architecture Overview

```mermaid
flowchart TD
    subgraph SRC["External Data Sources"]
        GBFS["GBFS API\nStation status & info\n(Velib Metropole)"]
        OWM["OpenWeather API 3.0\nCurrent + hourly forecast"]
        HOL["Python holidays\nPublic holiday calendar"]
    end

    subgraph INGEST["Data Ingestion  (Python 3.11)"]
        GI["gbfs_ingest.py\nevery 5 min"]
        WI["weather_ingest.py\nevery 1 hour"]
        HI["holidays_ingest.py\nyearly"]
    end

    subgraph PG["PostgreSQL 15  ·  public schema  (staging)"]
        STG1["stg_station_status\nstg_station_information"]
        STG2["stg_weather_current\nstg_weather_hourly"]
        STG3["stg_holidays"]
    end

    subgraph DBT["dbt 1.10  ·  analytics schema  (transformation)"]
        direction TB
        INT["Intermediate\nint_station_status_enriched\nint_station_neighbors (K=5, 0.8 km)\nint_station_rollups"]
        DIM["Dimensions\ndim_station  dim_date\ndim_time  dim_weather"]
        FCT["Fact\nfct_station_status (5-min grain)"]
        FEAT_OFF["Feature Store — Offline\nfeat_station_snapshot_5min\n33 features + labels\n(y_stockout_bikes/docks_30)"]
        FEAT_ON["Feature Store — Online\nfeat_station_snapshot_latest\n33 features, latest per station"]
    end

    subgraph TRAIN["Offline Training  (Local workstation)"]
        TR["train.py\nXGBoost / LightGBM\nTemporal split 80/20 + 60-min gap\nF-Beta threshold optimisation"]
        MLF["MLflow 3.10.1\nExperiment tracking\nModel registry (SQLite local)"]
        PKG["Model Package\npackage_manifest.json\n+ mlflow pyfunc model\n+ eval_summary.json"]
    end

    subgraph DEPLOY["Deployment State"]
        DS["model_dir/deployments/\nbikes/{local,staging,production}.json\ndocks/{local,staging,production}.json"]
    end

    subgraph SERVE["AWS Serving"]
        SM_S["SageMaker\nstaging-bikes  staging-docks"]
        SM_P["SageMaker\nproduction-bikes  production-docks"]
        LMB["Lambda Router\nresolves target + environment"]
        S3P["S3 — Predictions\nParquet shards\npartitioned by target/city/date"]
    end

    subgraph MON["Monitoring & Observability"]
        QB["quality_backfill.py\n30-min maturity window\nPR-AUC · F1 · Precision · Recall"]
        PSI["publish_psi.py\nPopulation Stability Index\nfeature baseline vs recent"]
        CW["AWS CloudWatch\nCustom metrics + Alarms\nPredictionHeartbeat"]
        APP["Streamlit Dashboard\napp/dashboard.py\nMap · Risk table · Model health"]
    end

    subgraph ORCH["Orchestration  (Apache Airflow 2.10.3 · CeleryExecutor)"]
        T1["Tier-1 Worker (concurrency=2)\nHotpath jobs\n• gbfs_ingestion_dag (5 min)\n• dbt_station_status_hotpath_dag\n• serving DAGs (15 min)"]
        T2["Tier-2 Worker (concurrency=1)\nBatch jobs\n• weather / holiday ingestion\n• dbt_feature_build_dag (1 hr)\n• quality, PSI, retraining"]
        RDS["Redis\nCelery message broker"]
    end

    GBFS --> GI --> STG1
    OWM --> WI --> STG2
    HOL --> HI --> STG3

    STG1 & STG2 & STG3 --> INT --> DIM & FCT
    FCT --> FEAT_OFF
    FCT --> FEAT_ON

    FEAT_OFF --> TR --> MLF
    TR --> PKG --> DS

    DS --> SM_S & SM_P
    LMB --> SM_S & SM_P

    FEAT_ON --> LMB
    SM_P --> S3P

    S3P --> QB --> CW
    FEAT_ON --> PSI --> CW
    CW --> APP

    T1 & T2 <--> RDS
    T1 --> GI & DBT & LMB
    T2 --> HI & DBT
```

### Diagram 2 — Data Pipeline Flow

```mermaid
flowchart LR
    subgraph APIs["External APIs"]
        A1["GBFS\n(5 min)"]
        A2["OpenWeather\n(1 hr)"]
        A3["Holidays\n(yearly)"]
    end

    subgraph STG["Postgres · public\nStaging (raw)"]
        S1["stg_station_status\nstg_station_information"]
        S2["stg_weather_current\nstg_weather_hourly"]
        S3["stg_holidays"]
    end

    subgraph INT2["dbt · analytics\nIntermediate (enriched)"]
        I1["int_station_status_enriched\ncapacity-aligned bike/dock counts"]
        I2["int_station_neighbors\nK=5 nearest within 0.8 km\ndistance-weighted"]
        I3["int_station_rollups\n15/30/60-min rolling windows"]
        I4["int_station_weather_aligned\nweather joined at 5-min grain"]
    end

    subgraph DIM2["dbt · analytics\nDimensions & Facts"]
        D1["dim_station (SCD2)\ndim_date  dim_time\ndim_weather"]
        F1["fct_station_status\n5-min grain\nall Paris stations"]
    end

    subgraph FS["dbt · analytics\nFeature Store"]
        FS1["feat_station_snapshot_5min\nOffline training\n33 features + binary labels"]
        FS2["feat_station_snapshot_latest\nOnline serving\n33 features, latest row per station"]
    end

    A1 --> S1
    A2 --> S2
    A3 --> S3

    S1 --> I1 & I2
    I1 --> I3 & I4
    S2 --> D1
    S3 --> D1

    I2 & I3 & I4 --> F1
    D1 --> F1

    F1 --> FS1 & FS2
```

**33 feature columns per snapshot:**

| Category | Features |
|---|---|
| Temporal | `hour`, `dow`, `is_weekend`, `is_holiday` |
| Inventory | `util_bikes`, `util_docks`, `capacity`, `delta_bikes_5m`, `delta_docks_5m`, `minutes_since_prev_snapshot` |
| Rolling windows | `roll15/30/60_net_bikes`, `roll15/30/60_bikes_mean` |
| Neighbor signals | `nbr_bikes_weighted`, `nbr_docks_weighted`, `neighbor_count_within_radius` |
| Weather — current | `temp`, `humidity`, `wind`, `precip`, `weather_code` |
| Weather — forecast | `hourly_temp`, `hourly_humidity`, `hourly_wind`, `hourly_precip`, `hourly_weather_code`, `hourly_precip_prob` |

### Diagram 3 — ML Lifecycle

```mermaid
flowchart LR
    subgraph DATA["Feature Store"]
        FS_TR["feat_station_snapshot_5min\n(Postgres · analytics)"]
    end

    subgraph TRAINING["Offline Training  src/training/train.py"]
        SP["Temporal Split\n80% train | 20% val\n60-min gap (no leakage)"]
        XGB["XGBoost / LightGBM\n500 estimators · max_depth=6\nlr=0.05 · subsample=0.8"]
        THR["Threshold Optimisation\nF-Beta on val set\nfavours recall"]
        EVL["Evaluation\nPR-AUC · F1 · Precision · Recall"]
    end

    subgraph MLFLOW["MLflow 3.10.1"]
        EXP["Experiment run\nparams · metrics · figures"]
        REG["Local registry\nSQLite backend\nmodel_dir/mlflow.db"]
    end

    subgraph PKG2["Model Package  model_dir/packages/"]
        MAN["package_manifest.json\ntarget · feature_order · threshold\nmlflow_run_id · versions"]
        MDL["mlflow pyfunc model\n(serialised XGBoost/LGB)"]
        EVAL2["eval_summary.json\nmodel_card.md"]
    end

    subgraph DEPST["Deployment State  model_dir/deployments/"]
        ENV["bikes/staging.json\nbikes/production.json\ndocks/staging.json\ndocks/production.json"]
    end

    subgraph SERVING2["AWS Serving"]
        EP_S["SageMaker\nstaging endpoints"]
        EP_P["SageMaker\nproduction endpoints"]
        PRED["Batch predictions\nevery 15 min\nS3 Parquet"]
    end

    subgraph MONITOR2["Monitoring  (30-min maturity window)"]
        QB2["Quality Backfill\nPR-AUC · F1 per 5-min bucket"]
        PSI2["PSI Drift\nbaseline vs recent features"]
        CW2["CloudWatch\ncustom metrics + alarms"]
    end

    FS_TR --> SP --> XGB --> THR --> EVL
    XGB --> EXP --> REG
    EVL --> MAN & MDL & EVAL2

    MAN & MDL --> ENV
    ENV --> EP_S & EP_P

    EP_P --> PRED --> QB2
    QB2 --> CW2
    PSI2 --> CW2

    CW2 -.->|drift alarm triggers| TRAINING
```

### Diagram 4 — Infrastructure & Deployment Topology

```mermaid
flowchart TD
    subgraph LOCAL["Local Workstation"]
        direction LR
        VENV["Python .venv\noffline training\nunit tests"]
        DC_L["Docker Compose\nlocal overrides\nAWS SSO credentials"]
    end

    subgraph GITHUB["GitHub"]
        REPO["Source repository"]
        CI["GitHub Actions CI\nruff · pytest · bandit\nterraform validate\nAWS OIDC smoke test"]
    end

    subgraph EC2["AWS EC2 (always-on)"]
        direction TB
        DC_E["Docker Compose\n(no local overrides)"]
        subgraph AIRFLOW_E["Airflow 2.10.3 · CeleryExecutor"]
            WEB["Webserver :8080"]
            SCH["Scheduler"]
            W1["Worker Tier-1\nconcurrency=2\nhotpath + serving"]
            W2["Worker Tier-2\nconcurrency=1\nbatch + quality"]
        end
        REDIS_E["Redis\nCelery broker"]
        PG_E["PostgreSQL 15\ndata warehouse +\nAirflow metadata"]
        DASH["Streamlit Dashboard"]
    end

    subgraph AWS["AWS Cloud Services"]
        direction TB
        subgraph SM["SageMaker"]
            EP_STG["staging-bikes\nstaging-docks"]
            EP_PRD["production-bikes\nproduction-docks"]
        end
        LAMBDA["Lambda\nrouter_lambda.py"]
        S3["S3 Bucket\n/raw  /predictions\n/quality  /psi"]
        CW3["CloudWatch\nmetrics + alarms"]
        SNS["SNS\nalert notifications"]
        ECR["ECR\ncontainer images"]
        TF["Terraform 1.13.3\ninfra/terraform/live/"]
    end

    LOCAL -->|git push| GITHUB
    GITHUB -->|CI validates| REPO
    REPO -->|deploy code| EC2

    EC2 <-->|SageMaker invoke| SM
    EC2 -->|write predictions| S3
    EC2 -->|publish metrics| CW3
    CW3 -->|alarm| SNS

    LAMBDA --> SM
    TF -.->|provisions| SM & S3 & CW3 & SNS & ECR & LAMBDA
```

## Data Flow
1. Python ingestion writes raw payloads and `public.stg_*`.
2. dbt builds curated and feature tables in `analytics.*`.
3. Offline training reads `analytics.feat_station_snapshot_5min`.
4. Online and batch inference read `analytics.feat_station_snapshot_latest`.
5. Training emits a package directory under `model_dir/packages/<target>/...`.
6. Activation writes a deployment-state record under `model_dir/deployments/<target>/<environment>.json`.

## Targets And Isolation
The platform supports two prediction targets on a shared code path:
- `bikes`
- `docks`

The following resources must remain target-specific:
- package roots
- deployment state
- S3 inference and monitoring partitions
- CloudWatch metric dimensions
- SageMaker endpoint names
- dashboard labels and queries

## Model Package
Each model package has this fixed structure:

```text
<package_dir>/
  package_manifest.json
  model/
  artifacts/
```

`package_manifest.json` is the static source of truth for:
- target definition
- threshold
- feature contract version
- feature order
- model identity
- registry metadata

## Deployment State
Deployment-state JSON is the dynamic source of truth for:
- active environment
- active package directory
- active registered model/version
- last activation timestamp
- active endpoint name when deployed

The formal layout is:

```text
model_dir/deployments/bikes/local.json
model_dir/deployments/bikes/staging.json
model_dir/deployments/bikes/production.json
model_dir/deployments/docks/local.json
model_dir/deployments/docks/staging.json
model_dir/deployments/docks/production.json
```

Single-file deployment state is legacy-only and must not be used for formal dual-target workflows.

## Runtime Contract
Formal runtime settings:
- Postgres: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`
- Scope: `CITY`, `BUCKET`, `TARGET_NAME`, `SERVING_ENVIRONMENT`
- Deployment: `DEPLOYMENT_STATE_PATH`, `MODEL_PACKAGE_DIR`
- MLflow local default: `MLFLOW_TRACKING_URI=sqlite:///model_dir/mlflow.db`

## Operating Split
Local workstation:
- code changes
- unit and integration tests
- offline training and model debug

EC2:
- Docker Compose
- Airflow scheduler/webserver
- Postgres
- dbt jobs
- serving DAGs for prediction, quality backfill, metrics publish, and PSI publish
- dashboard service

AWS:
- ECR
- S3
- IAM
- SageMaker staging/prod endpoints
- CloudWatch alarms and dashboards
- SNS notifications
- router lambda
- promote and rollback scripts
