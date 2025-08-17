# output "account_id" { value = data.aws_caller_identity.me.account_id }
# output "arn" { value = data.aws_caller_identity.me.arn }

output "bucket_name" { value = aws_s3_bucket.data.bucket }
output "glue_database" { value = aws_glue_catalog_database.db.name }
output "ecr_repository_url" { value = aws_ecr_repository.repo.repository_url }
output "lambda_function_name" { value = aws_lambda_function.placeholder.function_name }
output "sagemaker_role_arn" { value = aws_iam_role.sagemaker_exec.arn }
