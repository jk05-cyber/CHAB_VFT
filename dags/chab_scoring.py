import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/home/jk/chab-mlops")
from src import score

with DAG(
    dag_id="chab_batch_scoring",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # "@monthly" en production
    catchup=False,
    tags=["chab", "scoring"],
) as dag:
    PythonOperator(task_id="score_population",
                   python_callable=score.main, op_args=["2026-06-01"])
