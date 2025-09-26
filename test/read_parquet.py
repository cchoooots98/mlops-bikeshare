import pandas as pd

path = "s3://mlops-bikeshare-387706002632-ca-central-1/inference/city=nyc/dt=2025-09-26-03-40/predictions.parquet"
# read a single parquet file from S3 (make sure your AWS creds/SSO are configured)
df = pd.read_parquet(path, storage_options={"anon": False, "profile": "Shirley"})
print(df.head())
print(df.columns)
unique_prefixes = (
    df["inferenceId"]
    .astype("string")  # 安全转成可空字符串，不会把 NaN 变成 "nan"
    .str[:16]  # 取前16位
    .dropna()  # 去掉缺失
    .unique()  # 去重（保持出现顺序）
    .tolist()
)
print(unique_prefixes)

# read a partitioned directory from S3
# df_all = pd.read_parquet("s3://mlops-bikeshare-387706002632-ca-central-1/monitoring/quality/")  # reads all partitions
