# Alarm 1: Lambda errors
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${local.project}-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Lambda function has errors"
  dimensions          = { FunctionName = aws_lambda_function.placeholder.function_name }
  treat_missing_data  = "notBreaching"
}

# Alarm 2: SageMaker endpoint 5XX (created only when endpoint name is provided)
resource "aws_cloudwatch_metric_alarm" "sm_5xx" {
  count               = var.sagemaker_endpoint_name == "" ? 0 : 1
  alarm_name          = "${local.project}-sagemaker-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocation5XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "SageMaker endpoint returned 5XX errors"
  dimensions          = { EndpointName = var.sagemaker_endpoint_name }
  treat_missing_data  = "notBreaching"
}
