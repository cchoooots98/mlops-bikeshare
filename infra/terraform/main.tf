# S3 data lake bucket
resource "aws_s3_bucket" "data" {
  bucket = local.data_bucket_name
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

# Create the five top-level prefixes (as 0-byte objects)
resource "aws_s3_object" "prefixes" {
  for_each = toset(local.s3_prefixes)
  bucket   = aws_s3_bucket.data.id
  key      = each.value
  content  = ""
}

# Glue database (tables will come later after data lands)
resource "aws_glue_catalog_database" "db" {
  name = local.glue_db_name
}

# ECR repository for training/ inference images
resource "aws_ecr_repository" "repo" {
  name                 = local.ecr_repo_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}


# data "aws_caller_identity" "me" {}
data "aws_caller_identity" "current" {}