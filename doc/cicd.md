
# CI/CD Pipeline and Environment Definition

## 1. CI/CD Pipeline Overview

This MLOps project uses **GitHub Actions + AWS** for a fully automated workflow from training → registration → deployment.

### End-to-End Flow

1. **Development (dev)**
   - Local training/debugging → push code to feature branch  
   - GitHub Actions runs unit tests, linting, dependency checks  
   - Optional: small-scale training run for smoke testing  
   - Output: verified runnable code and MLflow experiment artifacts  

2. **Staging Deployment (stg)**
   - On merge to the `main` branch, CI automatically triggers **Step 5B**:  
     - Build Docker image → push to ECR  
     - Export latest MLflow Staging model → upload to S3  
     - Create/Update SageMaker **staging endpoint** (`bikeshare-staging`)  
   - Used for integration testing, API testing, and load validation  

3. **Production Deployment (prod)**
   - Manually approved workflow (`promote_prod.yml`)  
   - Transition MLflow model from **Staging → Production**  
   - Deploy to SageMaker **prod endpoint** (`bikeshare-prod`)  
   - Provides stable SLA for real business calls  

---

## 2. Environment Definitions

| Environment | Characteristics | Purpose | Cost/Resources |
|-------------|----------------|---------|----------------|
| **dev** | Local / temporary experiments | Data prep, feature engineering, model debugging | Low, CPU/GPU on demand |
| **stg** | Cloud Staging Endpoint (`bikeshare-staging`) | Integration tests, CI auto-deploy, validate new model performance | Medium, usually 1× `ml.m5.large` |
| **prod** | Cloud Production Endpoint (`bikeshare-prod`) | Stable online service with SLA | Higher, can scale horizontally with Auto Scaling |

### Key Configuration Items

- **Model Registry (MLflow)**  
  - dev: `runs:/<RUN_ID>/model`  
  - stg: `models:/bikeshare_risk/Staging`  
  - prod: `models:/bikeshare_risk/Production`

- **Docker Image (ECR)**  
  - dev: local build, not pushed  
  - stg: `mlflow-pyfunc:<tag>-stg`  
  - prod: `mlflow-pyfunc:<tag>-prod`

- **S3 Model Storage**  
  - `s3://mlops-bikeshare-<account>-ca-central-1/sagemaker/models/bikeshare_risk/model.tar.gz`

- **SageMaker Endpoint**  
  - stg: `bikeshare-staging`  
  - prod: `bikeshare-prod`

---

## 3. Pipeline Files

### `cd_staging.yml`
- **Trigger**: push to `main`  
- **Steps**:  
  1. Build Docker image and push to ECR  
  2. Export MLflow Staging model and upload to S3  
  3. Run `deploy_via_sagemaker_sdk.py` to deploy endpoint `bikeshare-staging`

### `promote_prod.yml`
- **Trigger**: manual dispatch  
- **Steps**:  
  1. Transition MLflow model from Staging → Production  
  2. Build Docker image with `-prod` tag  
  3. Deploy to endpoint `bikeshare-prod`

---

## 4. Developer Guide

- **Train locally (dev)**  
  ```powershell
  python pipelines/train.py --city nyc --model-type xgboost --experiment dev
  ```

- **Register to MLflow**

  ```powershell
  python pipelines/register_model.py --run-id <RUN_ID> --model-name bikeshare_risk --stage Staging
  ```

- **Manual Staging Deployment** (debugging)

  ```powershell
  python pipelines/deploy_via_sagemaker_sdk.py `
    --endpoint-name bikeshare-staging `
    --role-arn arn:aws:iam::<account>:role/mlops-bikeshare-sagemaker-exec `
    --image-uri <ECR_IMAGE_TAG> `
    --model-data <S3_URI> `
    --instance-type ml.m5.large `
    --region ca-central-1
  ```



## 5. Future Extensions

* **Monitoring**: integrate CloudWatch + Prometheus for latency, error rates, CPU/memory usage
* **Auto Scaling**: configure prod endpoint with min 1, max N instances, scale by QPS
* **Blue/Green or Canary Deployment**: gradually shift traffic from old version to new version
* **Data Feedback Loop**: capture online predictions + ground truth, write back to S3 → trigger retraining pipeline


