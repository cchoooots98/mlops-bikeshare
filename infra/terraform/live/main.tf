module "stack" {
  source = "../modules/platform"

  env                  = "live"
  aws_region           = var.aws_region
  sagemaker_endpoints  = var.sagemaker_endpoints
  github_owner         = var.github_owner
  github_repo_name     = var.github_repo_name
  repo_name            = var.repo_name
  role_name            = var.role_name
  city                 = var.city
  alarm_email_endpoint = var.alarm_email_endpoint
}
