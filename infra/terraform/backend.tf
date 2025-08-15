terraform {
  backend "s3" {
    bucket         = "mlops-tfstate-387706002632-ca-central-1"  # 例：mlops-tfstate-123456789012-ca-central-1
    key            = "infra/terraform.tfstate"
    region         = "ca-central-1"                                # 例：ca-central-1
    dynamodb_table = "mlops-tflock"
    encrypt        = true
  }
}
