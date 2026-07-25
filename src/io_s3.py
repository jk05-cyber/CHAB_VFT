"""Helpers d'I/O du data lake : MinIO (S3) ou systeme de fichiers local.

L'API publique est identique quel que soit le backend :
    upload_file / read_csv / read_parquet / write_parquet / exists / ensure_buckets
"""
import shutil
from pathlib import Path

import pandas as pd

from src import config


# --------------------------------------------------------------------------
# Backend local
# --------------------------------------------------------------------------
def _local_path(bucket: str, key: str) -> Path:
    p = config.LOCAL_LAKE_DIR / bucket / key
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Backend MinIO
# --------------------------------------------------------------------------
def s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=config.MINIO_ENDPOINT,
        aws_access_key_id=config.MINIO_ACCESS_KEY,
        aws_secret_access_key=config.MINIO_SECRET_KEY,
    )


def ensure_buckets():
    """Cree les 3 buckets s'ils n'existent pas (no-op en backend local)."""
    if config.STORAGE_BACKEND != "minio":
        for b in (config.BUCKET_BRONZE, config.BUCKET_SILVER, config.BUCKET_GOLD):
            (config.LOCAL_LAKE_DIR / b).mkdir(parents=True, exist_ok=True)
        return
    cli = s3_client()
    existing = {b["Name"] for b in cli.list_buckets().get("Buckets", [])}
    for b in (config.BUCKET_BRONZE, config.BUCKET_SILVER, config.BUCKET_GOLD):
        if b not in existing:
            cli.create_bucket(Bucket=b)
            print(f"[minio] bucket cree : {b}")


# --------------------------------------------------------------------------
# API publique
# --------------------------------------------------------------------------
def upload_file(local_path, bucket: str, key: str):
    if config.STORAGE_BACKEND == "minio":
        s3_client().upload_file(str(local_path), bucket, key)
    else:
        shutil.copyfile(str(local_path), _local_path(bucket, key))
    print(f"[bronze] {local_path} -> {bucket}/{key}")


def read_csv(bucket: str, key: str, **kwargs) -> pd.DataFrame:
    if config.STORAGE_BACKEND == "minio":
        return pd.read_csv(
            f"s3://{bucket}/{key}",
            storage_options=config.STORAGE_OPTIONS,
            low_memory=False,
            **kwargs,
        )
    return pd.read_csv(_local_path(bucket, key), low_memory=False, **kwargs)


def read_parquet(bucket: str, key: str) -> pd.DataFrame:
    if config.STORAGE_BACKEND == "minio":
        return pd.read_parquet(
            f"s3://{bucket}/{key}", storage_options=config.STORAGE_OPTIONS
        )
    return pd.read_parquet(_local_path(bucket, key))


def write_parquet(df: pd.DataFrame, bucket: str, key: str):
    if config.STORAGE_BACKEND == "minio":
        df.to_parquet(
            f"s3://{bucket}/{key}", index=False, storage_options=config.STORAGE_OPTIONS
        )
    else:
        df.to_parquet(_local_path(bucket, key), index=False)
    print(f"[write] {bucket}/{key} ({len(df):,} lignes x {df.shape[1]} colonnes)")


def exists(bucket: str, key: str) -> bool:
    if config.STORAGE_BACKEND == "minio":
        import botocore

        try:
            s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False
    return (config.LOCAL_LAKE_DIR / bucket / key).exists()
