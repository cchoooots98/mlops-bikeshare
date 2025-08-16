terraform {
  backend "s3" {
    bucket         = "mlops-tfstate-387706002632-ca-central-1"
    key            = "infra/terraform.tfstate"
    region         = "ca-central-1"
    encrypt        = true
    use_lockfile =true
  }
}
