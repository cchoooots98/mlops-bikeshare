# Comment: Centralized config readers for endpoint/city/region.
import os

REGION = os.getenv("AWS_REGION", "eu-west-3")
CITY = os.getenv("CITY", "nyc")
SM_ENDPOINT = os.getenv("SM_ENDPOINT", "bikeshare-prod")
CW_NAMESPACE = os.getenv("CW_NS", "Bikeshare/Model")
