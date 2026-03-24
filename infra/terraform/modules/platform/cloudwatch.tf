locals {
  monitoring_email_endpoint = var.alarm_email_endpoint == null ? null : trimspace(var.alarm_email_endpoint)
}

resource "aws_sns_topic" "monitoring" {
  name = "${local.project}-monitoring"
}

resource "aws_sns_topic_subscription" "monitoring_email" {
  count     = local.monitoring_email_endpoint == null || local.monitoring_email_endpoint == "" ? 0 : 1
  topic_arn = aws_sns_topic.monitoring.arn
  protocol  = "email"
  endpoint  = local.monitoring_email_endpoint
}

locals {
  endpoint_alarm_context = {
    for key, endpoint in var.sagemaker_endpoints :
    key => {
      endpoint_name = endpoint
      target_name   = startswith(key, "docks") ? "docks" : "bikes"
      environment   = endswith(key, "prod") ? "production" : "staging"
      short_label   = "${startswith(key, "docks") ? "docks" : "bikes"}-${endswith(key, "prod") ? "prod" : "staging"}"
    }
  }

  dashboard_pr_auc_metrics = [
    for ctx in values(local.endpoint_alarm_context) : [
      "Bikeshare/Model",
      "PR-AUC-24h",
      "Environment",
      ctx.environment,
      "EndpointName",
      ctx.endpoint_name,
      "City",
      var.city,
      "TargetName",
      ctx.target_name,
      { label = "${ctx.short_label} PR-AUC", region = var.aws_region },
    ]
  ]

  dashboard_f1_metrics = [
    for ctx in values(local.endpoint_alarm_context) : [
      "Bikeshare/Model",
      "F1-24h",
      "Environment",
      ctx.environment,
      "EndpointName",
      ctx.endpoint_name,
      "City",
      var.city,
      "TargetName",
      ctx.target_name,
      { label = "${ctx.short_label} F1", region = var.aws_region },
    ]
  ]

  dashboard_heartbeat_metrics = [
    for ctx in values(local.endpoint_alarm_context) : [
      "Bikeshare/Model",
      "PredictionHeartbeat",
      "Environment",
      ctx.environment,
      "EndpointName",
      ctx.endpoint_name,
      "City",
      var.city,
      "TargetName",
      ctx.target_name,
      { label = "${ctx.short_label} Heartbeat", region = var.aws_region },
    ]
  ]

  dashboard_psi_metrics = [
    for ctx in values(local.endpoint_alarm_context) : [
      "Bikeshare/Model",
      "PSI",
      "Environment",
      ctx.environment,
      "EndpointName",
      ctx.endpoint_name,
      "City",
      var.city,
      "TargetName",
      ctx.target_name,
      { label = "${ctx.short_label} PSI", region = var.aws_region },
    ]
  ]

  dashboard_psi_core_metrics = [
    for ctx in values(local.endpoint_alarm_context) : [
      "Bikeshare/Model",
      "PSI_core",
      "Environment",
      ctx.environment,
      "EndpointName",
      ctx.endpoint_name,
      "City",
      var.city,
      "TargetName",
      ctx.target_name,
      { label = "${ctx.short_label} PSI core", region = var.aws_region },
    ]
  ]

  dashboard_psi_weather_metrics = [
    for ctx in values(local.endpoint_alarm_context) : [
      "Bikeshare/Model",
      "PSI_weather",
      "Environment",
      ctx.environment,
      "EndpointName",
      ctx.endpoint_name,
      "City",
      var.city,
      "TargetName",
      ctx.target_name,
      { label = "${ctx.short_label} PSI weather", region = var.aws_region },
    ]
  ]

  dashboard_latency_metrics = [
    for ctx in values(local.endpoint_alarm_context) : [
      "AWS/SageMaker",
      "ModelLatency",
      "EndpointName",
      ctx.endpoint_name,
      "VariantName",
      "AllTraffic",
      { label = "${ctx.short_label} Latency p95", region = var.aws_region, stat = "p95" },
    ]
  ]

  dashboard_5xx_metrics = [
    for ctx in values(local.endpoint_alarm_context) : [
      "AWS/SageMaker",
      "Invocation5XXErrors",
      "EndpointName",
      ctx.endpoint_name,
      "VariantName",
      "AllTraffic",
      { label = "${ctx.short_label} 5XX", region = var.aws_region, stat = "Sum" },
    ]
  ]
}

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${local.project}-router-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Router Lambda has errors"
  dimensions          = { FunctionName = aws_lambda_function.router.function_name }
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.monitoring.arn]
}

resource "aws_cloudwatch_metric_alarm" "sm_5xx" {
  for_each            = local.endpoint_alarm_context
  alarm_name          = "${local.project}-${each.value.endpoint_name}-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Invocation5XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "SageMaker endpoint returned 5XX errors"
  dimensions          = { EndpointName = each.value.endpoint_name, VariantName = "AllTraffic" }
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.monitoring.arn]
}

resource "aws_cloudwatch_metric_alarm" "sm_latency_p95" {
  for_each            = local.endpoint_alarm_context
  alarm_name          = "${local.project}-${each.value.endpoint_name}-latency-p95"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 300
  extended_statistic  = "p95"
  threshold           = 200000
  alarm_description   = "SageMaker endpoint p95 latency is above 200 ms"
  dimensions          = { EndpointName = each.value.endpoint_name, VariantName = "AllTraffic" }
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.monitoring.arn]
}

resource "aws_cloudwatch_metric_alarm" "quality_pr_auc_low" {
  for_each            = local.endpoint_alarm_context
  alarm_name          = "${local.project}-${each.value.endpoint_name}-pr-auc-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "PR-AUC-24h"
  namespace           = "Bikeshare/Model"
  period              = 900
  statistic           = "Average"
  threshold           = 0.70
  alarm_description   = "Target-aware PR-AUC over the last window is below the admission threshold"
  dimensions = {
    Environment  = each.value.environment
    EndpointName = each.value.endpoint_name
    City         = var.city
    TargetName   = each.value.target_name
  }
  treat_missing_data = "notBreaching"
  alarm_actions      = [aws_sns_topic.monitoring.arn]
}

resource "aws_cloudwatch_metric_alarm" "quality_f1_low" {
  for_each            = local.endpoint_alarm_context
  alarm_name          = "${local.project}-${each.value.endpoint_name}-f1-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "F1-24h"
  namespace           = "Bikeshare/Model"
  period              = 900
  statistic           = "Average"
  threshold           = 0.55
  alarm_description   = "Target-aware F1 over the last window is below the admission threshold"
  dimensions = {
    Environment  = each.value.environment
    EndpointName = each.value.endpoint_name
    City         = var.city
    TargetName   = each.value.target_name
  }
  treat_missing_data = "notBreaching"
  alarm_actions      = [aws_sns_topic.monitoring.arn]
}

resource "aws_cloudwatch_metric_alarm" "quality_heartbeat_low" {
  for_each            = local.endpoint_alarm_context
  alarm_name          = "${local.project}-${each.value.endpoint_name}-heartbeat-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "PredictionHeartbeat"
  namespace           = "Bikeshare/Model"
  period              = 900
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Target-aware prediction heartbeat is missing for the latest 15-minute window"
  dimensions = {
    Environment  = each.value.environment
    EndpointName = each.value.endpoint_name
    City         = var.city
    TargetName   = each.value.target_name
  }
  treat_missing_data = "breaching"
  alarm_actions      = [aws_sns_topic.monitoring.arn]
}

resource "aws_cloudwatch_metric_alarm" "quality_psi_high" {
  for_each            = local.endpoint_alarm_context
  alarm_name          = "${local.project}-${each.value.endpoint_name}-psi-core-high"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "PSI_core"
  namespace           = "Bikeshare/Model"
  period              = 900
  statistic           = "Maximum"
  threshold           = 0.20
  alarm_description   = "Target-aware core-feature PSI has crossed the warning threshold"
  dimensions = {
    Environment  = each.value.environment
    EndpointName = each.value.endpoint_name
    City         = var.city
    TargetName   = each.value.target_name
  }
  treat_missing_data = "notBreaching"
  alarm_actions      = [aws_sns_topic.monitoring.arn]
}

resource "aws_cloudwatch_dashboard" "monitoring" {
  dashboard_name = "${local.project}-${var.env}-monitoring"
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 2
        properties = {
          markdown = "# Bikeshare Monitoring Dashboard\nFour formal SageMaker endpoints, target-aware quality metrics, and alert-oriented service health."
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 2
        width  = 12
        height = 6
        properties = {
          title   = "PR-AUC-24h / F1-24h"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = concat(local.dashboard_pr_auc_metrics, local.dashboard_f1_metrics)
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 2
        width  = 12
        height = 6
        properties = {
          title   = "PredictionHeartbeat / PSI Signals"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = concat(
            local.dashboard_heartbeat_metrics,
            local.dashboard_psi_metrics,
            local.dashboard_psi_core_metrics,
            local.dashboard_psi_weather_metrics,
          )
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 8
        width  = 12
        height = 6
        properties = {
          title   = "ModelLatency p95"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = local.dashboard_latency_metrics
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 8
        width  = 12
        height = 6
        properties = {
          title   = "Invocation5XXErrors"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = local.dashboard_5xx_metrics
        }
      },
    ]
  })
}
