import pandas as pd
path = "s3://mlops-bikeshare-387706002632-ca-central-1/monitoring/quality/city=nyc/ds=2025-09-25/part-2025-09-25-01-55.parquet"
# read a single parquet file from S3 (make sure your AWS creds/SSO are configured)
df = pd.read_parquet(path, storage_options={"anon": False, "profile": "Shirley"})
print(df.head())
print(df.columns)

# read a partitioned directory from S3
# df_all = pd.read_parquet("s3://mlops-bikeshare-387706002632-ca-central-1/monitoring/quality/")  # reads all partitions
