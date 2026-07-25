"""Scoring batch : applique la derniere version du modele sur la population S2."""
import os
import mlflow
import pandas as pd
from src import config
from src.io_s3 import read_parquet, write_parquet

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", config.MINIO_ENDPOINT)


def main(snapshot_date: str = None):
    snapshot_date = snapshot_date or config.SNAPSHOTS["S2"]
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model("models:/chab_appetence/latest")

    df = read_parquet(config.BUCKET_GOLD,
                      f"scoring/snapshot_date={snapshot_date}/scoring_population.parquet")
    X = df.drop(columns=[c for c in ["client_id", "snapshot_id", "snapshot_date"]
                         if c in df.columns])
    df_out = df[["client_id"]].copy()
    df_out["score_appetence_chab"] = model.predict_proba(X)[:, 1]
    df_out["decile"] = pd.qcut(df_out["score_appetence_chab"].rank(method="first"),
                               10, labels=range(10, 0, -1)).astype(int)
    df_out = df_out.sort_values("score_appetence_chab", ascending=False)
    write_parquet(df_out, config.BUCKET_GOLD,
                  f"scores/snapshot_date={snapshot_date}/scores_chab.parquet")
    print(df_out.head(10))


if __name__ == "__main__":
    main()
