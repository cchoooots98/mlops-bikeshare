provider "aws" {
  region = var.aws_region
   default_tags {
    tags = {
      Project     = var.repo_name
      Environment = var.env
      ManagedBy   = "terraform"
    }
  }
}
