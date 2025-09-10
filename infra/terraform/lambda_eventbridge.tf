data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "${local.project}-lambda-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "archive_file" "lambda_zip" {
  type        = "zip"
  output_path = "${path.module}/lambda.zip"

  source {
    filename = "main.py"
    content  = <<PY
def handler(event, context):
    return {"status": "ok", "msg": "placeholder"}
PY
  }
}

resource "aws_lambda_function" "placeholder" {
  function_name = local.lambda_function_name
  role          = aws_iam_role.lambda_exec.arn
  handler       = "main.handler"
  runtime       = "python3.12"
  filename      = data.archive_file.lambda_zip.output_path
  timeout       = 30
  environment {
    variables = {
      PROJECT      = local.project
      BUCKET       = aws_s3_bucket.data.bucket
      CW_NAMESPACE = local.cw_namespace
    }
  }
}

resource "aws_cloudwatch_event_rule" "every_15min" {
  name                = "${local.project}-schedule-15min"
  schedule_expression = "rate(15 minutes)"
  state               = "DISABLED"
}

resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.every_15min.name
  target_id = "lambda"
  arn       = aws_lambda_function.placeholder.arn
}

resource "aws_lambda_permission" "event_invoke" {
  statement_id  = "AlllowExecutionFromEventBridge"
  action        = "lambda:InvokeFuction"
  function_name = aws_lambda_function.placeholder.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.every_15min.arn
}