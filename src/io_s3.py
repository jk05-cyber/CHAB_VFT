"""Helpers de lecture/écriture MinIO (S3) en CSV/Parquet."""
import boto3
import pandas as pd
from src import config


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=config.MINIO_ENDPOINT,
        aws_access_key_id=config.MINIO_ACCESS_KEY,
        aws_secret_access_key=config.MINIO_SECRET_KEY,
    )


def upload_file(local_path: str, bucket: str, key: str):
    s3_client().upload_file(str(local_path), bucket, key)
    print(f"[bronze] {local_path} -> s3://{bucket}/{key}")


def read_csv(bucket: str, key: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(f"s3://{bucket}/{key}",
                       storage_options=config.STORAGE_OPTIONS,
                       low_memory=False, **kwargs)


def read_parquet(bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(f"s3://{bucket}/{key}",
                           storage_options=config.STORAGE_OPTIONS)


def write_parquet(df: pd.DataFrame, bucket: str, key: str):
    df.to_parquet(f"s3://{bucket}/{key}", index=False,
                  storage_options=config.STORAGE_OPTIONS)
    print(f"[write] s3://{bucket}/{key} ({len(df):,} lignes)")
