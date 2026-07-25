import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/home/jk/chab-mlops")  # remplace par ton chemin

from src import bronze, silver, gold, train

S1, S2 = "2026-01-01", "2026-06-01"

default_args = {"owner": "mlops", "retries": 1}

with DAG(
    dag_id="chab_e2e_medallion_training",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # declenchement manuel ; mettre "@monthly" en prod
    catchup=False,
    default_args=default_args,
    tags=["chab", "medallion", "training"],
) as dag:

    ingest = PythonOperator(task_id="bronze_ingest", python_callable=bronze.ingest_all)

    silver_tasks = []
    for snap in (S1, S2):
        for name, fn in (("rc", silver.clean_rc), ("pc", silver.clean_pc),
                         ("mvt", silver.clean_mvt)):
            silver_tasks.append(PythonOperator(
                task_id=f"silver_{name}_{snap.replace('-', '')}",
                python_callable=fn, op_args=[snap]))

    gold_feats = [PythonOperator(task_id=f"gold_features_{s.replace('-', '')}",
                                 python_callable=gold.build_features, op_args=[s])
                  for s in (S1, S2)]

    training_set = PythonOperator(task_id="gold_training_set",
                                  python_callable=gold.build_training_set)
    scoring_set = PythonOperator(task_id="gold_scoring_set",
                                 python_callable=gold.build_scoring_set)
    train_model = PythonOperator(task_id="train_register_model",
                                 python_callable=train.main)

    ingest >> silver_tasks
    for t in silver_tasks:
        t >> gold_feats
    gold_feats >> training_set >> train_model
    gold_feats >> scoring_set
