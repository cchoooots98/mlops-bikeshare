provider "aws" {
  region = var.aws_region
  # Optional. Prefer AWS_PROFILE or a dedicated caller identity for your account.
  profile = var.aws_profile
}
