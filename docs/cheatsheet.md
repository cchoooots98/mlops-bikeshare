# MLOps Project Cheatsheet

常用命令与操作速查（Windows + VS Code + GitHub + AWS）。

---

## VS Code

```bash
code .
Ctrl + `
exit
Shift + Alt + F
Ctrl + P
Ctrl + Shift + F
Ctrl + Shift + V
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
.venv\Scripts\activate
pip install -r requirements.txt
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
terraform apply
```

说明：
- `bootstrap` 只负责创建 tfstate bucket。
- `live` 使用 S3 native lockfile 处理 state locking。
- `live` 负责长期平台资源。
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
docker compose ps
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

## 数据与特征命令

```bash
python src/features/build_features.py --city paris --eda
python src/features/build_features.py --city paris --start "2025-08-18 00:00" --end "2025-08-22 00:00"
```
