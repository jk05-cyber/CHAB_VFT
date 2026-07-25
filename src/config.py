import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
BUCKET_BRONZE = os.environ["BUCKET_BRONZE"]
BUCKET_SILVER = os.environ["BUCKET_SILVER"]
BUCKET_GOLD = os.environ["BUCKET_GOLD"]
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
LANDING_DIR = Path(os.environ["LANDING_DIR"])

SNAPSHOTS = {"S1": "2026-01-01", "S2": "2026-06-01"}

STORAGE_OPTIONS = {
    "key": MINIO_ACCESS_KEY,
    "secret": MINIO_SECRET_KEY,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
}
