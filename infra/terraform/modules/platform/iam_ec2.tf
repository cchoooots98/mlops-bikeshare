# ── EC2 Instance Role ─────────────────────────────────────────────────
# Grants the EC2 host (running Docker Compose + Streamlit dashboard) the
# minimum permissions it needs:
#   • S3 read  — list bucket + get objects (predictions, quality Parquets)
#   • CloudWatch read — GetMetricData + ListMetrics (dashboard tabs 3 & 4)
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

# S3 read-only — dashboard reads predictions and quality Parquets
data "aws_iam_policy_document" "ec2_s3_read" {
  statement {
    sid       = "ListProjectBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.data.arn]
  }
  statement {
    sid       = "ReadProjectObjects"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
}

resource "aws_iam_policy" "ec2_s3_read" {
  name   = "${local.project}-ec2-s3-read"
  policy = data.aws_iam_policy_document.ec2_s3_read.json
}

# CloudWatch read — dashboard fetches metric time series for tabs 3 & 4
data "aws_iam_policy_document" "ec2_cw_read" {
  statement {
    sid       = "ReadCloudWatchMetrics"
    actions   = ["cloudwatch:GetMetricData", "cloudwatch:ListMetrics"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "ec2_cw_read" {
  name   = "${local.project}-ec2-cw-read"
  policy = data.aws_iam_policy_document.ec2_cw_read.json
}

resource "aws_iam_role_policy_attachment" "ec2_attach_s3" {
  role       = aws_iam_role.ec2.name
  policy_arn = aws_iam_policy.ec2_s3_read.arn
}

resource "aws_iam_role_policy_attachment" "ec2_attach_cw" {
  role       = aws_iam_role.ec2.name
  policy_arn = aws_iam_policy.ec2_cw_read.arn
}
