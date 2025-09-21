# create_monitoring_schedules.py
# Purpose: Create DataQuality + ModelQuality job definitions and hourly schedules.
# Run:  python create_monitoring_schedules.py

import boto3
import botocore
from sagemaker import image_uris

region = "ca-central-1"
bucket = "mlops-bikeshare-387706002632-ca-central-1"  # EDIT
endpoint_name = "bikeshare-staging"  # EDIT
role_arn = "arn:aws:iam::387706002632:role/mlops-bikeshare-sagemaker-exec"  # EDIT

baseline_constraints = f"s3://{bucket}/monitoring/baseline/city=nyc/constraints.json"
baseline_statistics = f"s3://{bucket}/monitoring/baseline/city=nyc/statistics.json"
reports_prefix = f"s3://{bucket}/monitoring/reports/"
groundtruth_prefix = f"s3://{bucket}/monitoring/quality/city=nyc"
image_uri = image_uris.retrieve(framework="model-monitor", region=region)
# image_uri = "536280801234.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-model-monitor-analyzer"

preprocessor_uri = f"s3://{bucket}/monitoring/code/record_preprocessor.py"

sm = boto3.client("sagemaker", region_name=region)

# 1) DataQuality job definition (covers schema, nulls, ranges, AND drift vs baseline)
dq_name = "bikeshare-data-quality-jd"
try:
    sm.create_data_quality_job_definition(
        JobDefinitionName=dq_name,
        DataQualityAppSpecification={"ImageUri": image_uri, "RecordPreprocessorSourceUri": preprocessor_uri},
        DataQualityBaselineConfig={
            "ConstraintsResource": {"S3Uri": baseline_constraints},
            "StatisticsResource": {"S3Uri": baseline_statistics},
        },
        DataQualityJobInput={
            "EndpointInput": {
                "EndpointName": endpoint_name,
                "LocalPath": "/opt/ml/processing/input_data",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            }
        },
        DataQualityJobOutputConfig={
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
        NetworkConfig={"EnableNetworkIsolation": False, "EnableInterContainerTrafficEncryption": False},
        RoleArn=role_arn,
        StoppingCondition={"MaxRuntimeInSeconds": 3300},
    )
    print(f"Job Definition is created: {dq_name} ")
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "ResourceInUse":
        print(f"Job Definition already exists: {dq_name} (reusing)")
    else:
        raise

# 2) ModelQuality job definition (computes classification metrics vs ground truth)
mq_post_uri = f"s3://{bucket}/monitoring/code/mq_postprocessor.py"
mq_name = "bikeshare-model-quality-jd"
try:
    sm.create_model_quality_job_definition(
        JobDefinitionName=mq_name,
        ModelQualityAppSpecification={
            "ImageUri": image_uri,
            "ProblemType": "BinaryClassification",
            "PostAnalyticsProcessorSourceUri": mq_post_uri,
        },
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


# 3) Create THREE schedules (hourly)
try:
    sm.create_monitoring_schedule(
        MonitoringScheduleName="bikeshare-data-quality",
        MonitoringScheduleConfig={
            "ScheduleConfig": {"ScheduleExpression": "cron(0 * ? * * *)"},
            "MonitoringJobDefinitionName": dq_name,
            "MonitoringType": "DataQuality",
        },
    )
    print("Schedule already exists: bikeshare-data-quality")
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "ResourceInUse":
        print("Schedule already exists: bikeshare-data-quality (reusing)")
    else:
        raise
try:
    sm.create_monitoring_schedule(
        MonitoringScheduleName="bikeshare-data-drift",
        MonitoringScheduleConfig={
            "ScheduleConfig": {"ScheduleExpression": "cron(0 * ? * * *)"},
            "MonitoringJobDefinitionName": dq_name,
            "MonitoringType": "DataQuality",
        },
    )
    print("Schedule already exists: bikeshare-data-drift")
except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "ResourceInUse":
        print("Schedule already exists: bikeshare-data-drift (reusing)")
    else:
        raise

try:
    sm.create_monitoring_schedule(
        MonitoringScheduleName="bikeshare-model-quality",
        MonitoringScheduleConfig={
            "ScheduleConfig": {
                "ScheduleExpression": "NOW",
                "DataAnalysisStartTime": "-PT5H",
                "DataAnalysisEndTime": "-PT0H",
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

print("All job definitions and schedules created.")
