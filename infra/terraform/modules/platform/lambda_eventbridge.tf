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

resource "aws_iam_role_policy_attachment" "lambda_sagemaker_invoke" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

data "archive_file" "router_lambda_zip" {
  type        = "zip"
  output_path = "${path.module}/router_lambda.zip"
  source_dir  = "${path.module}/router_lambda_src"
}

resource "aws_lambda_function" "router" {
  function_name    = local.lambda_function_name
  role             = aws_iam_role.lambda_exec.arn
  handler          = "main.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.router_lambda_zip.output_path
  source_code_hash = data.archive_file.router_lambda_zip.output_base64sha256
  timeout          = 30
  environment {
    variables = merge(
      {
        PROJECT             = local.project
        BUCKET              = aws_s3_bucket.data.bucket
        CW_NAMESPACE        = local.cw_namespace
        CW_NS               = local.cw_namespace
        DEFAULT_ENVIRONMENT = "production"
      },
      {
        for key, value in var.sagemaker_endpoints :
        "ENDPOINT_${upper(replace(key, "-", "_"))}" => value
      }
    )
  }
}
