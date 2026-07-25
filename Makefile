.PHONY: help venv install data bronze silver gold stats train score all clean docker-up docker-down mlflow airflow lint

PY := .venv/bin/python
PIP := .venv/bin/pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Cree l'environnement virtuel
	python3 -m venv .venv

install: venv ## Installe les dependances
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt

data: ## Genere les donnees synthetiques dans data/landing
	$(PY) scripts/generate_synthetic_landing.py --clients 20000

bronze: ## Ingestion Bronze
	$(PY) -m src.bronze

silver: ## Nettoyage Silver
	$(PY) -m src.silver

gold: ## Feature engineering + cible + population de scoring
	$(PY) -m src.gold

stats: ## Analyses statistiques (khi2, Cramer V, Mann-Whitney, IV, VIF, PSI, forward)
	$(PY) -m src.stats_analysis

train: ## Benchmark multi-modeles (RL, RF, XGBoost, LightGBM, CatBoost)
	$(PY) -m src.train

train-fast: ## Benchmark rapide
	$(PY) -m src.train --fast --no-cv

score: ## Scoring batch avec le champion
	$(PY) -m src.score

all: data bronze silver gold stats train score ## Pipeline complet end-to-end

mlflow: ## Lance l'UI MLflow (http://localhost:5000)
	.venv/bin/mlflow ui --backend-store-uri $${MLFLOW_TRACKING_URI:-sqlite:///mlflow/mlflow.db} --host 0.0.0.0 --port 5000

docker-up: ## Lance MinIO + Postgres + MLflow
	docker compose up -d

docker-down: ## Arrete la stack Docker
	docker compose down

clean: ## Supprime lake, artefacts et caches
	rm -rf data/lake artifacts/* __pycache__ src/__pycache__ .pytest_cache
