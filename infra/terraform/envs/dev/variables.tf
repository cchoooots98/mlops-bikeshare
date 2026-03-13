variable "aws_region" {
  type    = string
  default = "eu-west-3"
}

variable "aws_profile" {
  type    = string
  default = null
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
  type    = string
  default = "cchoooots98"
}

variable "repo_name" {
  type    = string
  default = "mlops-bikeshare"
}

variable "role_name" {
  type    = string
  default = "gh-oidc-deployer"
}

variable "city" {
  type    = string
  default = "paris"
}

variable "alarm_email_endpoint" {
  type    = string
  default = null
}
