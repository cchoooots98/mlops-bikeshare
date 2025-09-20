# Resolve the official Model Monitor analyzer image for your current AWS Region
# Works on Windows + VS Code + PowerShell

from sagemaker import Session, image_uris

# Get active region from your AWS config/credentials
region = Session().boto_region_name or "ca-central-1"

# Ask SageMaker SDK for the right image URI for Model Monitor
# This returns "<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-model-monitor-analyzer:<tag>"
mm_image = image_uris.retrieve(framework="model-monitor", region=region)

print("Resolved Model Monitor image for region:", region)
print(mm_image)

# Example use in your code:
# DataQualityAppSpecification={"ImageUri": mm_image}
# ModelQualityAppSpecification={"ImageUri": mm_image, "ProblemType": "BinaryClassification"}
