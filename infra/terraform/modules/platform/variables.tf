variable "aws_region" {
  type    = string
  default = "eu-west-3"
}

variable "sagemaker_endpoints" {
  type = map(string)
  default = {
    bikes_staging = "bikeshare-bikes-staging"
    bikes_prod    = "bikeshare-bikes-prod"
    docks_staging = "bikeshare-docks-staging"
    docks_prod    = "bikeshare-docks-prod"
  }
}

variable "github_owner" {
  type        = string
  description = "GitHub org/user that owns the repo"
  default     = "cchoooots98"
}

variable "repo_name" {
  type        = string
  description = "Project/resource prefix used for AWS resources"
  default     = "mlops-bikeshare"
}

variable "github_repo_name" {
  type        = string
  description = "GitHub repository name allowed to assume the OIDC role"
  default     = null
}

variable "role_name" {
  type        = string
  description = "IAM role name for GitHub OIDC"
  default     = "gh-oidc-deployer"
}

variable "env" {
  type    = string
  default = "live"
}

variable "city" {
  type        = string
  description = "City dimension used for target-aware custom metrics and dashboards"
  default     = "paris"
}

variable "alarm_email_endpoint" {
  type        = string
  description = "Optional email address to subscribe to the monitoring SNS topic"
  default     = null
}

locals {
  project                  = var.repo_name
  account_id               = data.aws_caller_identity.current.account_id
  github_repo_subject_name = coalesce(var.github_repo_name, var.repo_name)
  # Data lake bucket: <repo>-<account>-<region>
  data_bucket_name     = "${var.repo_name}-${local.account_id}-${var.aws_region}"
  cw_namespace         = "Bikeshare/Model"
  glue_db_name         = replace(var.repo_name, "-", "_")
  ecr_repo_name        = var.repo_name
  lambda_function_name = "${var.repo_name}-router"
  s3_prefixes          = ["raw/", "curated/", "features/", "inference/", "monitoring/"]

  default_tags = {
    Project     = var.repo_name
    Environment = var.env
    ManagedBy   = "terraform"
  }
}
