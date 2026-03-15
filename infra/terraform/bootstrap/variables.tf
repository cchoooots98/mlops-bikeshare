variable "aws_region" {
  type    = string
  default = "eu-west-3"
}

variable "aws_profile" {
  type    = string
  default = null
}

variable "repo_name" {
  type    = string
  default = "mlops-bikeshare"
}

variable "tf_state_bucket_name" {
  type    = string
  default = null
}

variable "tf_lock_table_name" {
  type    = string
  default = null
}
