variable "aws_region" {
  type    = string
  default = "ca-central-1"
}

variable "aws_profile" {
  type    = string
  default = "Shirley"
}

variable "sagemaker_endpoint_name" {
  type    = string
  default = "" # keep empty until you actually deploy an endpoint
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


locals {
  project    = var.repo_name
  account_id = data.aws_caller_identity.current.account_id
  # Data lake bucket: <repo>-<account>-<region>
  data_bucket_name     = "${var.repo_name}-${local.account_id}-${var.aws_region}"
  cw_namespace         = "MLOps/Bikeshare"
  glue_db_name         = replace(var.repo_name, "-", "_")
  ecr_repo_name        = var.repo_name
  lambda_function_name = "${var.repo_name}-placeholder"
  s3_prefixes          = ["raw/", "curated/", "features/", "inference/", "monitoring/"]

  default_tags = {
    Project     = var.repo_name
    Environment = var.env
    ManagedBy   = "terraform"
  }
}