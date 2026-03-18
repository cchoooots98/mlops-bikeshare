import pandas as pd

path = "s3://bikeshare-paris-387706002632-eu-west-3/monitoring/quality/city=paris/ds=2025-09-30/part-2025-09-30-05-40.parquet"
# read a single parquet file from S3 (make sure your AWS creds/SSO are configured)
df = pd.read_parquet(path, storage_options={"anon": False, "profile": "Shirley-fr"})
print(df.head())
print(df.columns)


# read a partitioned directory from S3
# df_all = pd.read_parquet("s3://bikeshare-paris-387706002632-eu-west-3/monitoring/quality/")  # reads all partitions
