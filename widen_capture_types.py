# widen_capture_types_ec.py
# Goal: Widen Data Capture JsonContentTypes for an existing endpoint
# without creating a new model (avoid 63-char model name limit).

import os, datetime, boto3, copy

# Region and endpoint name come from env or fall back to your defaults
REGION   = os.environ.get("AWS_REGION", "ca-central-1")     # your AWS region
ENDPOINT = os.environ.get("SM_ENDPOINT", "bikeshare-staging")  # target endpoint

sm = boto3.client("sagemaker", region_name=REGION)

# 1) Get the live endpoint and its endpoint-config name
ep = sm.describe_endpoint(EndpointName=ENDPOINT)  # describe current endpoint
old_epc_name = ep["EndpointConfigName"]           # current endpoint-config name

# 2) Read the endpoint-config to reuse model/variant/capture settings
old_epc = sm.describe_endpoint_config(EndpointConfigName=old_epc_name)
pv = old_epc["ProductionVariants"][0]             # assume single variant
variant_name = pv.get("VariantName", "AllTraffic")
model_name   = pv["ModelName"]                    # reuse the same model
instance_type  = pv["InstanceType"]
instance_count = pv["InitialInstanceCount"]
capture_conf = copy.deepcopy(old_epc.get("DataCaptureConfig", {}))

# 3) Widen JsonContentTypes so responses like "application/json; charset=utf-8" are captured
dcch = capture_conf.get("CaptureContentTypeHeader", {})
json_types = set((dcch.get("JsonContentTypes") or []))
# Add common JSON MIME types that often appear in real deployments
json_types.update([
    "application/json",
    "application/json; charset=utf-8",
    "application/json;charset=utf-8",
    "text/json",
    "application/jsonlines",
    "application/jsonl",
])
# Ensure the minimal structure exists
capture_conf["EnableCapture"] = True                          # keep capture enabled
capture_conf["InitialSamplingPercentage"] = max(1, capture_conf.get("InitialSamplingPercentage", 100))
capture_conf["CaptureOptions"] = [{"CaptureMode":"Input"},{"CaptureMode":"Output"}]
capture_conf["CaptureContentTypeHeader"] = {
    "JsonContentTypes": sorted(json_types),                   # widened JSON types
    "CsvContentTypes": dcch.get("CsvContentTypes") or ["text/csv"],  # keep CSV or set default
}

# 4) Create a new endpoint-config name (<=63 chars)
ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")     # UTC timestamp to avoid collisions
suffix = f"-captypes-{ts}"                                    # short suffix for clarity
# Truncate from the left if needed to keep <=63 chars (AWS hard limit)
base = (old_epc_name + suffix)[-63:]

new_epc_name = base

# 5) Create the new endpoint-config reusing the same model and instance settings
sm.create_endpoint_config(
    EndpointConfigName=new_epc_name,                          # new config name
    ProductionVariants=[{
        "VariantName": variant_name,
        "ModelName": model_name,                              # reuse existing model
        "InitialVariantWeight": 1.0,
        "InitialInstanceCount": instance_count,
        "InstanceType": instance_type
    }],
    DataCaptureConfig=capture_conf                            # widened capture types
)

# 6) Point the live endpoint to the new endpoint-config (rolling update)
sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=new_epc_name)

print(f"[ok] Updated {ENDPOINT} to {new_epc_name}")
print("[hint] New S3 prefix will be:")
print(f"datacapture/endpoint={new_epc_name}/{ENDPOINT}/AllTraffic/YYYY/MM/DD/HH/")
