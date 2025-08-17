data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "${local.project}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

# S3 access scoped to this project's bucket
data "aws_iam_policy_document" "sagemaker_s3" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
}

resource "aws_iam_policy" "sagemaker_s3" {
  name   = "${local.project}-sagemaker-s3"
  policy = data.aws_iam_policy_document.sagemaker_s3.json
}

# Logs, metrics, and ECR pull 
data "aws_iam_policy_document" "sagemaker_logs_ecr" {
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["*"]
  }
  statement {
    actions = [
      "ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"
    ]
    resources = ["*"]
  }
  statement {
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "sagemaker_logs_ecr" {
  name   = "${local.project}-sagemaker-logs-ecr"
  policy = data.aws_iam_policy_document.sagemaker_logs_ecr.json
}

resource "aws_iam_role_policy_attachment" "attach_s3" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = aws_iam_policy.sagemaker_s3.arn
}
resource "aws_iam_role_policy_attachment" "attach_logs_ecr" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = aws_iam_policy.sagemaker_logs_ecr.arn
}