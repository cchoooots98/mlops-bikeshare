from time import sleep

import boto3
from sagemaker import image_uris

region = "ca-central-1"
bucket = "mlops-bikeshare-387706002632-ca-central-1"
endpoint_name = "bikeshare-staging"
role_arn = "arn:aws:iam::387706002632:role/mlops-bikeshare-sagemaker-exec"

sm = boto3.client("sagemaker", region_name=region)


# def get_capture_prefix(endpoint):
#     ep = sm.describe_endpoint(EndpointName=endpoint)
#     return ep["DataCaptureConfig"]["DestinationS3Uri"]


reports_prefix = f"s3://{bucket}/monitoring/reports"
image_uri = image_uris.retrieve(framework="model-monitor", region=region)
print(f"Image URI: {image_uri}")
# capture_prefix = get_capture_prefix(endpoint_name) #
# Image URI: 536280801234.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-model-monitor-analyzer

try:
    sm.delete_monitoring_schedule(MonitoringScheduleName="bikeshare-model-quality")
    sleep(30)
except sm.exceptions.ResourceNotFound:
    pass

try:
    sm.delete_model_quality_job_definition(JobDefinitionName="bikeshare-model-quality-jd")
    sleep(30)
except sm.exceptions.ResourceNotFound:
    pass


sm.create_model_quality_job_definition(
    JobDefinitionName="bikeshare-model-quality-jd",
    ModelQualityAppSpecification={"ImageUri": image_uri, "ProblemType": "BinaryClassification"},
    ModelQualityJobInput={
        "EndpointInput": {
            "EndpointName": "bikeshare-staging",
            "LocalPath": "/opt/ml/processing/input_data",
            "S3InputMode": "File",
            "ProbabilityAttribute": "predictions",
            "ProbabilityThresholdAttribute": 0.15,
            "StartTimeOffset": "-PT8H",
            "EndTimeOffset": "-PT4H",
        },
        "GroundTruthS3Input": {"S3Uri": f"s3://{bucket}/monitoring/ground-truth"},
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
print("Job Definition is created: bikeshare-model-quality-jd")


sm.create_monitoring_schedule(
    MonitoringScheduleName="bikeshare-model-quality",
    MonitoringScheduleConfig={
        "ScheduleConfig": {
            "ScheduleExpression": "cron(0 0/2 ? * * *)",
        },
        "MonitoringJobDefinitionName": "bikeshare-model-quality-jd",
        "MonitoringType": "ModelQuality",
    },
)
print("Schedule already exists: bikeshare-model-quality")
