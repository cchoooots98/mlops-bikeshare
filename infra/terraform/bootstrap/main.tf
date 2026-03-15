data "aws_caller_identity" "current" {}

locals {
  state_bucket_name = coalesce(
    var.tf_state_bucket_name,
    "${var.repo_name}-tfstate-${data.aws_caller_identity.current.account_id}-${var.aws_region}"
  )

  default_tags = {
    Project   = var.repo_name
    ManagedBy = "terraform"
    Stack     = "bootstrap"
  }
}

resource "aws_s3_bucket" "tf_state" {
  bucket = local.state_bucket_name
  tags   = local.default_tags
}

resource "aws_s3_bucket_public_access_block" "tf_state" {
  bucket                  = aws_s3_bucket.tf_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
