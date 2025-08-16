

# GitHub OIDC provider
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# Assume role policy document
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${var.github_owner}/${var.repo_name}:ref:refs/heads/main",
        "repo:${var.github_owner}/${var.repo_name}:environment:staging",
        "repo:${var.github_owner}/${var.repo_name}:environment:prod",
      ]
    }

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
  }
}

# IAM Role for GitHub Actions
resource "aws_iam_role" "gh_deployer" {
  name                 = var.role_name
  assume_role_policy   = data.aws_iam_policy_document.assume.json
  max_session_duration = 3600
}

# 最小权限策略（单独资源，避免 inline_policy 警告）
data "aws_iam_policy_document" "least_priv" {
  statement {
    sid       = "STS"
    effect    = "Allow"
    actions   = ["sts:GetCallerIdentity"]
    resources = ["*"]
  }

  statement {
    sid       = "LogsCloudWatch"
    effect    = "Allow"
    actions   = ["logs:*", "events:*", "cloudwatch:*"]
    resources = ["*"]
  }

  statement {
    sid       = "S3Basic"
    effect    = "Allow"
    actions   = ["s3:*"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "gh_deployer_least_priv" {
  name   = "least-priv"
  role   = aws_iam_role.gh_deployer.id
  policy = data.aws_iam_policy_document.least_priv.json
}

# 输出

output "gh_oidc_role_arn" {
  value = aws_iam_role.gh_deployer.arn
}
