output "aws_region" {
  value = var.aws_region
}

output "tf_state_bucket_name" {
  value = aws_s3_bucket.tf_state.bucket
}
