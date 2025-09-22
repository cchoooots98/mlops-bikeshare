import boto3
import botocore
from sagemaker import image_uris

region = "ca-central-1"
bucket = "mlops-bikeshare-387706002632-ca-central-1"
endpoint_name = "bikeshare-staging"
role_arn = "arn:aws:iam::387706002632:role/mlops-bikeshare-sagemaker-exec"

sm = boto3.client("sagemaker", region_name=region)


def get_capture_prefix(endpoint):
    ep = sm.describe_endpoint(EndpointName=endpoint)
    return ep["DataCaptureConfig"]["DestinationS3Uri"]


groundtruth_prefix = "s3://mlops-bikeshare-387706002632-ca-central-1/monitoring/quality/city=nyc"
reports_prefix = f"s3://{bucket}/monitoring/reports"
image_uri = image_uris.retrieve(framework="model-monitor", region=region)
capture_prefix = get_capture_prefix(endpoint_name)

mq_name = "bikeshare-model-quality-jd"
try:
    sm.create_model_quality_job_definition(
        JobDefinitionName=mq_name,
        ModelQualityAppSpecification={"ImageUri": image_uri, "ProblemType": "BinaryClassification"},
        ModelQualityJobInput={
            "EndpointInput": {
                "EndpointName": endpoint_name,
                "LocalPath": "/opt/ml/processing/input_data",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
                "ProbabilityAttribute": "yhat",
                "ProbabilityThresholdAttribute": 0.15,
            },
            "GroundTruthS3Input": {"S3Uri": groundtruth_prefix},
        },
        ModelQualityJobOutputConfig={
            "MonitoringOutputs": [
                {
                    "S3Output": {
                        "S3Uri": reports_prefix,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    }
                }
            ]
        },
        JobResources={"ClusterConfig": {"InstanceCount": 1, "InstanceType": "ml.m5.large", "VolumeSizeInGB": 30}},
        NetworkConfig={"EnableNetworkIsolation": False},
        RoleArn=role_arn,
        StoppingCondition={"MaxRuntimeInSeconds": 3300},
    )
    print(f"Job Definition is created: {mq_name} ")
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "ResourceInUse":
        print(f"Job Definition already exists: {mq_name} (reusing)")
    else:
        raise

try:
    sm.create_monitoring_schedule(
        MonitoringScheduleName="bikeshare-model-quality",
        MonitoringScheduleConfig={
            "ScheduleConfig": {
                "ScheduleExpression": "NOW",
                "DataAnalysisStartTime": "-PT5H",
                "DataAnalysisEndTime": "-PT1H",
            },  # cron(0 0/2 ? * * *)
            "MonitoringJobDefinitionName": mq_name,
            "MonitoringType": "ModelQuality",
        },
    )
    print("Schedule already exists: bikeshare-model-quality")
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "ResourceInUse":
        print("Schedule already exists: bikeshare-model-quality (reusing)")
    else:
        raise
