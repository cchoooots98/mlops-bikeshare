terraform {
  backend "s3" {
    bucket         = "mlops-tfstate-387706002632-eu-west-3"
    key            = "infra/terraform.tfstate"
    region         = "eu-west-3"
    dynamodb_table = "mlops-tflock"
    encrypt        = true
  }
}
