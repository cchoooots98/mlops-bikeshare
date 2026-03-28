# MLOps Project Cheatsheet

常用命令与操作速查（Windows + VS Code + GitHub + AWS）。

---

## VS Code

```
code .              # 打开项目
Ctrl + `            # 打开终端
Shift + Alt + F     # 格式化代码
Ctrl + P            # 文件快速跳转
Ctrl + Shift + F    # 全局搜索
Ctrl + Shift + V    # 预览 Markdown
```

---

## Git / GitHub

### 初始化与仓库
```bash
git init
git clone <repo_url>
```

### 状态与历史
```bash
git status
git log --oneline --graph --all
```

### 分支操作
```bash
git branch
git checkout -b dev
git switch main
git merge dev
```

### 提交与同步
```bash
git add .
git commit -m "feat: add pipeline script"
git push origin main
git pull origin main
```

### GitHub CLI
```bash
gh auth login
gh repo create <name>
gh pr create --fill
gh pr status
gh issue list
```

---

## Python / 环境

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/Mac

pip install -r requirements.txt -r requirements-dev.txt
pip check
pytest -q
```

---

## AWS CLI

```bash
aws sso login --profile admin-sso
aws sts get-caller-identity
aws s3 ls
```

---

## Terraform

```bash
cd infra/terraform/bootstrap
terraform init
terraform apply

cd ../live
terraform init -reconfigure \
  -backend-config="bucket=<tfstate-bucket>" \
  -backend-config="key=infra/live/terraform.tfstate" \
  -backend-config="region=eu-west-3" \
  -backend-config="encrypt=true"
terraform plan
# 仅在 apply 因 EntityAlreadyExists 报错时才需要 import：
terraform import module.stack.aws_iam_openid_connect_provider.github arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com
terraform import module.stack.aws_iam_role.gh_deployer gh-oidc-deployer
terraform import module.stack.aws_iam_role_policy.gh_deployer_least_priv gh-oidc-deployer:least-priv
terraform apply
```

说明：
- `bootstrap` 只负责创建 tfstate bucket。
- `live` 使用 S3 native lockfile 处理 state locking。
- `live` 负责长期平台资源。
- 如果共享账号里已经存在 GitHub OIDC provider 或 `gh-oidc-deployer`，先 import 再 apply。
- `staging` / `production` 属于模型发布流程，不是 Terraform 双环境。

---

## 日常推荐流程

1. 开始工作
```bash
aws sso login --profile admin-sso
.venv\Scripts\activate
```

2. 开发与验证
```bash
pytest -q
docker compose -f docker-compose.yml -f docker-compose.local.yml ps
```

3. 提交与推送
```bash
git add .
git commit -m "fix: correct schema check"
git push origin main
```

4. 发布流程
- GitHub Actions 跑 CI
- 部署到 staging endpoint
- 通过 gate 后再 promote 到 production

---

## 离线训练

```powershell
# 训练 bikes 模型
.\.venv\Scripts\python.exe -m src.training.train `
  --city paris `
  --start "2026-03-01 00:00" `
  --end "2026-03-07 23:55" `
  --predict-bikes true `
  --model-type xgboost

# 训练 docks 模型
.\.venv\Scripts\python.exe -m src.training.train `
  --city paris `
  --start "2026-03-01 00:00" `
  --end "2026-03-07 23:55" `
  --predict-bikes false `
  --model-type xgboost
```

---

## 本地 Orchestration Loop（需要可达的 AWS 端点）

```powershell
# 运行推理预测器
$env:TARGET_NAME = "bikes"
$env:SERVING_ENVIRONMENT = "local"
$env:DEPLOYMENT_STATE_PATH = "model_dir/deployments/bikes/local.json"
.\.venv\Scripts\python.exe -m src.inference.predictor

# 运行质量回填
.\.venv\Scripts\python.exe -m src.monitoring.quality_backfill

# 运行指标 dry-run
.\.venv\Scripts\python.exe -m src.monitoring.metrics.publish_custom_metrics `
  --bucket bikeshare-paris-387706002632-eu-west-3 `
  --quality-prefix AUTO `
  --endpoint bikeshare-bikes-staging `
  --city-dimension paris `
  --target-name bikes `
  --environment staging `
  --dry-run
```

---

## 模型发布流程

```powershell
# 1. 导出 & 上传模型包
.\.venv\Scripts\python.exe -m pipelines.export_and_upload_model `
  --package-dir model_dir/packages/bikes/<run-dir> `
  --output-dir dist/model_packages `
  --s3-uri s3://<bucket>/packages/bikes/latest.tar.gz `
  --region eu-west-3

# 2. 部署 staging 端点
.\.venv\Scripts\python.exe -m pipelines.deploy_staging `
  --endpoint-name bikeshare-bikes-staging `
  --role-arn <role_arn> `
  --image-uri <image_uri> `
  --package-s3-uri s3://<bucket>/packages/bikes/latest.tar.gz `
  --package-dir model_dir/packages/bikes/<run-dir>

# 3. 运行 gate 检查（24小时观察窗口后）
.\.venv\Scripts\python.exe -m test.check_gate `
  --endpoint bikeshare-bikes-staging `
  --city paris --region eu-west-3 `
  --target-name bikes --environment staging

# 4. Promote 到生产
.\.venv\Scripts\python.exe -m pipelines.promote `
  --source-deployment-state-path model_dir/deployments/bikes/staging.json `
  --target-deployment-state-path model_dir/deployments/bikes/production.json `
  --target-environment production

# 5. 回滚（如需要）
.\.venv\Scripts\python.exe -m pipelines.rollback `
  --target-name bikes `
  --environment production `
  --from-state model_dir/deployments/bikes/production.json `
  --to-state model_dir/deployments/bikes/previous_prod.json
```

---

## Docker Compose（EC2 运维）

```bash
# 查看服务状态
docker compose ps

# 查看日志
docker compose logs airflow-webserver --tail 100
docker compose logs airflow-worker-core --tail 50

# 重启 Airflow 服务
docker compose up -d --build --force-recreate \
  airflow-webserver airflow-scheduler \
  airflow-worker-core airflow-worker-weather airflow-worker-serving airflow-worker-obs airflow-worker-psi airflow-worker-sidecar

# 检查 DAG 导入错误
docker compose exec airflow-webserver airflow dags list-import-errors

# Smoke test 预测任务
docker compose exec airflow-webserver airflow tasks test \
  staging_prediction_15min predict_bikes <logical-date>
```

---

## Streamlit Dashboard

```powershell
# 安装依赖
pip install -r requirements-app.txt

# 本地运行
streamlit run app/dashboard.py
```
