provider "aws" {
  region = var.aws_region
}
variable "aws_region" {
  type    = string
  default = "ca-central-1"
}
