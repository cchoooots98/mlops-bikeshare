variable "aws_region" {
  type    = string
  default = "ca-central-1"
}

variable "github_owner" {
  type        = string
  description = "GitHub org/user that owns the repo"
}

variable "repo_name" {
  type        = string
  description = "GitHub repository name"
}

variable "role_name" {
  type        = string
  description = "IAM role name for GitHub OIDC"
  default     = "gh-oidc-deployer"
}