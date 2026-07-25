"""Bronze : copie brute et immuable des CSV vers MinIO, partitionnée par snapshot."""
from src import config
from src.io_s3 import upload_file

FILES = ["Vue_RC.csv", "Vue_PC.csv", "Vue_MVT.csv"]


def ingest_snapshot(snapshot_date: str):
    src_dir = config.LANDING_DIR / f"snapshot_{snapshot_date}"
    for f in FILES:
        local = src_dir / f
        if not local.exists():
            raise FileNotFoundError(local)
        upload_file(local, config.BUCKET_BRONZE,
                    f"snapshot_date={snapshot_date}/{f}")


def ingest_all():
    for d in config.SNAPSHOTS.values():
        ingest_snapshot(d)


if __name__ == "__main__":
    ingest_all()
