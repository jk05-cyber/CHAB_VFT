"""Entrainement du score d'appetence CHAB avec tracking et registry MLflow."""
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src import config
from src.io_s3 import read_parquet

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", config.MINIO_ENDPOINT)

CAT_COLS = ["business_line", "profil_1", "profil_2", "profil_3", "segment",
            "marche", "csp", "ville", "sexe", "type_personne", "pack"]
DROP_COLS = ["client_id", "target", "snapshot_id", "snapshot_date"]


def lift_at_decile(y_true, y_score, decile=1):
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False)
    top = df.head(int(len(df) * decile / 10))
    return top["y"].mean() / df["y"].mean()


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("chab_appetence")

    df = read_parquet(config.BUCKET_GOLD, "training/chab_training_set.parquet")
    y = df["target"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="INCONNU")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
        ]), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])

    spw = (y == 0).sum() / (y == 1).sum()
    model = Pipeline([
        ("pre", pre),
        ("clf", LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, random_state=42, n_jobs=-1)),
    ])

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    with mlflow.start_run(run_name="lgbm_chab"):
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        metrics = {
            "roc_auc": roc_auc_score(y_va, p),
            "pr_auc": average_precision_score(y_va, p),
            "lift_decile_1": lift_at_decile(y_va.values, p),
        }
        mlflow.log_params({"scale_pos_weight": round(spw, 2),
                           "n_features": X.shape[1], "n_train": len(X_tr)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model",
                                 registered_model_name="chab_appetence")
        print(metrics)


if __name__ == "__main__":
    main()
