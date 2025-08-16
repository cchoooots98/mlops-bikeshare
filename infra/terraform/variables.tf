variable "aws_region" {
  type    = string
  default = "ca-central-1"
}

variable "github_owner" {
  type        = string
  description = "GitHub org/user that owns the repo"
  default     = "cchoooots98"
}

variable "repo_name" {
  type        = string
  description = "GitHub repository name"
  default     = "mlops-bikeshare"
}

variable "role_name" {
  type        = string
  description = "IAM role name for GitHub OIDC"
  default     = "gh-oidc-deployer"
}

variable "env" {
  type    = string
  default = "dev"
}

# 下面三项虽然当前 backend.tf 没有引用，但保留做参数化也可以。
variable "tf_state_bucket" {
  type    = string
  default = "mlops-tfstate-387706002632-ca-central-1"
}

variable "tf_lock_table" {
  type    = string
  default = "mlops-tflock"
}

variable "tf_state_key" {
  type    = string
  default = "infra/terraform.tfstate"
}
