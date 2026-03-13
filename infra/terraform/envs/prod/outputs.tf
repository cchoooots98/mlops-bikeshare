output "data_bucket_name" {
  value = module.stack.data_bucket_name
}

output "router_lambda_name" {
  value = module.stack.router_lambda_name
}

output "sns_topic_arn" {
  value = module.stack.sns_topic_arn
}

output "cloudwatch_dashboard_name" {
  value = module.stack.cloudwatch_dashboard_name
}
