import boto3

region = "ca-central-1"
job_def_name = "bikeshare-model-quality-jd"

sm = boto3.client("sagemaker", region_name=region)

# 获取 ModelQuality Job Definition 的输入 S3 URI
resp = sm.describe_model_quality_job_definition(JobDefinitionName=job_def_name)
inputs = resp["ModelQualityJobInput"]["BatchTransformInput"]
print("Input S3 URI:", inputs["DataCapturedDestinationS3Uri"])
print("DatasetFormat:", inputs["DatasetFormat"])
print("LocalPath:", inputs["LocalPath"])
print("ProbabilityAttribute:", inputs.get("ProbabilityAttribute"))
print("ProbabilityThresholdAttribute:", inputs.get("ProbabilityThresholdAttribute"))
print("StartTimeOffset:", inputs.get("StartTimeOffset"))
print("EndTimeOffset:", inputs.get("EndTimeOffset"))

# 获取最近一次监控任务的详细日志
schedules = sm.list_monitoring_schedules(MonitoringJobDefinitionName=job_def_name)
if schedules["MonitoringScheduleSummaries"]:
    schedule_name = schedules["MonitoringScheduleSummaries"][0]["MonitoringScheduleName"]
    executions = sm.list_monitoring_executions(
        MonitoringScheduleName=schedule_name, MaxResults=1, SortOrder="Descending"
    )
    if executions["MonitoringExecutionSummaries"]:
        exec_summary = executions["MonitoringExecutionSummaries"][0]
        print("Execution summary:")
        for k, v in exec_summary.items():
            print(f"{k}: {v}")
    else:
        print("No executions found.")
else:
    print("No schedules found.")
