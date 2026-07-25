"""Configuration centrale du projet CHAB_VFT (score d'appetence Credit Habitat).

Toutes les variables sensibles sont lues depuis .env (jamais commitees).
Deux backends de stockage sont supportes :
  - minio : buckets S3 (production / demo Docker)
  - local : dossiers sur disque (dev rapide, aucun service a lancer)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default=None, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and val in (None, ""):
        raise RuntimeError(f"Variable d'environnement manquante : {key} (voir .env.example)")
    return val


# --------------------------------------------------------------------------
# Stockage
# --------------------------------------------------------------------------
STORAGE_BACKEND = _env("STORAGE_BACKEND", "local").lower()  # local | minio
LOCAL_LAKE_DIR = Path(_env("LOCAL_LAKE_DIR", str(PROJECT_ROOT / "data" / "lake")))

MINIO_ENDPOINT = _env("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = _env("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = _env("MINIO_SECRET_KEY", "minioadmin")

BUCKET_BRONZE = _env("BUCKET_BRONZE", "chab-bronze")
BUCKET_SILVER = _env("BUCKET_SILVER", "chab-silver")
BUCKET_GOLD = _env("BUCKET_GOLD", "chab-gold")

STORAGE_OPTIONS = {
    "key": MINIO_ACCESS_KEY,
    "secret": MINIO_SECRET_KEY,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
}

# --------------------------------------------------------------------------
# Donnees
# --------------------------------------------------------------------------
LANDING_DIR = Path(_env("LANDING_DIR", str(PROJECT_ROOT / "data" / "landing")))

# Snapshot S1 = photo des features ; snapshot S2 = observation de la cible
SNAPSHOTS = {"S1": _env("SNAPSHOT_S1", "2026-01-01"), "S2": _env("SNAPSHOT_S2", "2026-06-01")}

# --------------------------------------------------------------------------
# MLflow / modelisation
# --------------------------------------------------------------------------
MLFLOW_TRACKING_URI = _env("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow/mlflow.db")
EXPERIMENT_NAME = _env("MLFLOW_EXPERIMENT", "chab_appetence")
REGISTERED_MODEL = _env("MLFLOW_REGISTERED_MODEL", "chab_appetence")

RANDOM_STATE = int(_env("RANDOM_STATE", "42"))
TEST_SIZE = float(_env("TEST_SIZE", "0.2"))
CV_FOLDS = int(_env("CV_FOLDS", "5"))
# Metrique de selection du champion : pr_auc | roc_auc | lift_decile_1
CHAMPION_METRIC = _env("CHAMPION_METRIC", "pr_auc")

ARTIFACTS_DIR = Path(_env("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    return (
        f"backend={STORAGE_BACKEND} | landing={LANDING_DIR} | snapshots={SNAPSHOTS} | "
        f"mlflow={MLFLOW_TRACKING_URI} | champion_metric={CHAMPION_METRIC}"
    )


if __name__ == "__main__":
    print(summary())
