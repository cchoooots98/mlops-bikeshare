# EC2 instance role.
# Grants the EC2 host (running Docker Compose + the dashboard) the minimum
# permissions it needs:
# - S3 bucket/object access for reading and writing project artifacts
# - CloudWatch metric read/write access for monitoring and custom metrics
# - SageMaker invoke/describe access limited to known staging/prod endpoints
#
# If the role was created manually under a different name, import it first:
#   terraform import aws_iam_role.ec2 <existing-role-name>
#   terraform import aws_iam_instance_profile.ec2 <existing-profile-name>

data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ec2" {
  name               = "${local.project}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${local.project}-ec2-profile"
  role = aws_iam_role.ec2.name
}

# S3 project bucket access. Airflow serving writes predictions/quality shards,
# the dashboard reads them back, and operators may clean up bad artifacts.
data "aws_iam_policy_document" "ec2_s3_read" {
  statement {
    sid       = "ListProjectBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.data.arn]
  }

  statement {
    sid       = "ManageProjectObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
}

resource "aws_iam_policy" "ec2_s3_read" {
  name   = "${local.project}-ec2-s3-read"
  policy = data.aws_iam_policy_document.ec2_s3_read.json
}

# CloudWatch access. Serving publishes custom metrics and the dashboard reads
# monitoring series.
data "aws_iam_policy_document" "ec2_cw_read" {
  statement {
    sid = "ReadAndPublishCloudWatchMetrics"
    actions = [
      "cloudwatch:GetMetricData",
      "cloudwatch:ListMetrics",
      "cloudwatch:PutMetricData",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "ec2_cw_read" {
  name   = "${local.project}-ec2-cw-read"
  policy = data.aws_iam_policy_document.ec2_cw_read.json
}

data "aws_iam_policy_document" "ec2_sagemaker_invoke" {
  statement {
    sid = "InvokeAndDescribeKnownSageMakerEndpoints"
    actions = [
      "sagemaker:DescribeEndpoint",
      "sagemaker:InvokeEndpoint",
    ]
    resources = [
      for endpoint_name in values(var.sagemaker_endpoints) :
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:endpoint/${endpoint_name}"
    ]
  }
}

resource "aws_iam_policy" "ec2_sagemaker_invoke" {
  name   = "${local.project}-ec2-sagemaker-invoke"
  policy = data.aws_iam_policy_document.ec2_sagemaker_invoke.json
}

resource "aws_iam_role_policy_attachment" "ec2_attach_s3" {
  role       = aws_iam_role.ec2.name
  policy_arn = aws_iam_policy.ec2_s3_read.arn
}

resource "aws_iam_role_policy_attachment" "ec2_attach_cw" {
  role       = aws_iam_role.ec2.name
  policy_arn = aws_iam_policy.ec2_cw_read.arn
}

resource "aws_iam_role_policy_attachment" "ec2_attach_sagemaker" {
  role       = aws_iam_role.ec2.name
  policy_arn = aws_iam_policy.ec2_sagemaker_invoke.arn
}
