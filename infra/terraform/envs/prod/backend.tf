terraform {
  backend "s3" {
    # Replace these values if they do not match your real remote state backend.
    bucket         = "mlops-tfstate-387706002632-eu-west-3"
    key            = "infra/prod/terraform.tfstate"
    region         = "eu-west-3"
    dynamodb_table = "mlops-tflock"
    encrypt        = true
  }
}
