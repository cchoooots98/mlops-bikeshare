# 🚀 MLOps Project Cheatsheet

常用命令与操作速查（Windows + VS Code + GitHub + AWS）。

---

## 🖥 VS Code 常用操作

```bash
# 打开 VS Code 当前目录
code .

# 打开/关闭终端
Ctrl + `         # 打开内置终端
exit             # 关闭终端

# 格式化代码
Shift + Alt + F

# 快速搜索文件
Ctrl + P

# 搜索文件内容
Ctrl + Shift + F

# Markdown 预览
Ctrl + Shift + V
```

---

## 🌱 Git / GitHub

### 初始化与仓库
```bash
git init                       # 初始化 git 仓库
git clone <repo_url>           # 克隆远程仓库
```

### 查看状态与历史
```bash
git status                     # 查看改动
git log --oneline --graph --all  # 历史记录图
```

### 分支操作
```bash
git branch                     # 查看分支
git checkout -b dev            # 新建并切换 dev 分支
git switch main                # 切换回 main
git merge dev                  # 合并 dev 分支
```

### 提交改动
```bash
git add .                      # 添加所有改动
git commit -m "feat: add pipeline script"
```

### 同步远程
```bash
git push origin main           # 推送 main 分支
git pull origin main           # 拉取更新
```

### GitHub CLI
```bash
gh auth login                  # 登录 GitHub
gh repo create <name>          # 创建仓库
gh pr create --fill            # 创建 Pull Request
gh pr status                   # 查看 PR 状态
gh issue list                  # 列出 issues
```

---

## 🐍 Python / 环境

```bash
python -m venv .venv           # 创建虚拟环境
.venv\Scripts\activate         # 激活虚拟环境 (Windows)
pip install -r requirements.txt  # 安装依赖
pytest -q                      # 运行测试
```

---

## 🌩 AWS CLI

```bash
aws sso login --profile admin-sso     # 登录 AWS
aws sts get-caller-identity           # 验证身份
aws s3 ls                             # 列出 S3 桶
```

---

## 🌍 Terraform

```bash
cd infra/terraform
terraform init                        # 初始化
terraform plan                        # 查看执行计划
terraform apply                       # 应用更改
```

---

## 📌 日常工作流程（推荐）

1. **开机/开始工作**
   ```bash
   aws sso login --profile admin-sso   # 登录 AWS
   .venv\Scripts\activate              # 激活 Python 环境
   ```

2. **开发**
   - 修改代码  
   - 本地运行 & 测试：`pytest -q`

3. **提交与推送**
   ```bash
   git add .
   git commit -m "fix: correct schema check"
   git push origin main
   ```

4. **CI/CD**
   - GitHub Actions 自动运行 CI → 部署到 staging  
   - 审核后手动 promote 到 prod  

---

# 最近14天 + 报告
python src/features/build_features.py --city paris --eda

# 只回填窗口
python src/features/build_features.py --city paris --start "2025-08-18 00:00" --end "2025-08-22 00:00"

# Athena 修分区
