# Walkthrough Ubuntu — CHAB_VFT end-to-end

Guide pas-a-pas, **100 % copiable**, testé sur Ubuntu 22.04 / 24.04.
Deux parcours : **A. local rapide** (aucun service) et **B. stack complète** (MinIO + MLflow + Airflow).

---

## Étape 0 — Prérequis système

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential git curl make \
                    libgomp1 unzip jq
python3 --version   # >= 3.10
```

> `libgomp1` est indispensable à LightGBM / XGBoost / CatBoost (OpenMP).

---

## Étape 1 — Récupérer le dépôt

```bash
cd ~
git clone https://github.com/jk05-cyber/CHAB_VFT.git chab-mlops
cd chab-mlops
git checkout feature/multi-modeles
```

---

## Étape 2 — Environnement Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

Vérification :

```bash
python -c "import sklearn, lightgbm, xgboost, catboost, mlflow, scipy; print('OK')"
```

---

## Étape 3 — Configuration

```bash
cp .env.example .env
nano .env       # laisse STORAGE_BACKEND=local pour le parcours A
python -m src.config     # affiche le résumé de configuration
```

---

# Parcours A — Exécution locale (5 minutes, aucun service)

## A.1 Générer les données

```bash
python scripts/generate_synthetic_landing.py --clients 20000
tree -L 2 data/landing | head -20
```

> Remplace cette étape par tes vrais extraits si tu les as :
> `data/landing/snapshot_2026-01-01/{Vue_RC.csv,Vue_PC.csv,Vue_MVT.csv}`

## A.2 Bronze → Silver → Gold

```bash
python -m src.bronze     # copie brute immuable
python -m src.silver     # nettoyage, typage, dédup, winsorisation
python -m src.gold       # features + cible S1→S2 + population de scoring
```

Contrôle :

```bash
find data/lake -name "*.parquet" | sort
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/lake/chab-gold/training/chab_training_set.parquet")
print(df.shape, "| taux positifs :", round(df.target.mean()*100, 2), "%")
PY
```

## A.3 Analyses statistiques

```bash
python -m src.stats_analysis
ls -1 artifacts/stats/
```

Produit :

| Fichier | Contenu |
|---|---|
| `profil_dataset.json` | volumétrie, taux de positifs, ratio de déséquilibre |
| `missing_rates.csv` | taux de valeurs manquantes par variable |
| `stats_numeriques.csv` | Mann-Whitney U, point-biserial, KS, Cohen's d, IV |
| `stats_categorielles.csv` | Khi-deux, ddl, **Cramér's V**, IV |
| `correlation_spearman.csv` / `correlations_fortes.csv` | multicolinéarité |
| `vif.csv` | Variance Inflation Factor |
| `psi_s1_vs_s2.csv` | **PSI** (drift) entre S1 et S2 |
| `forward_selection.csv` | variables retenues + gain d'AUC par étape |

Lecture rapide :

```bash
column -s, -t artifacts/stats/stats_numeriques.csv | head -15
column -s, -t artifacts/stats/stats_categorielles.csv | head -15
cat artifacts/stats/profil_dataset.json | jq
```

## A.4 Benchmark multi-modèles

```bash
# rapide (dégrossir)
python -m src.train --fast --no-cv

# complet : RL, RF, XGBoost, LightGBM, CatBoost + CV 5 folds + calibration
python -m src.train

# variantes
python -m src.train --models xgboost,lightgbm,catboost
python -m src.train --tune            # RandomizedSearchCV
python -m src.train --sample 50000    # sous-échantillonnage
```

Sorties :

```bash
column -s, -t artifacts/models/comparatif_modeles.csv | less -S
cat artifacts/models/comparatif_modeles.md
column -s, -t artifacts/models/tests_statistiques.csv
ls artifacts/models/figures/
```

## A.5 Scoring batch

```bash
python -m src.score
head -5 artifacts/scoring/top_clients_2026-06-01.csv
```

## A.6 Tout en une commande

```bash
make all
```

## A.7 Visualiser MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 0.0.0.0 --port 5000
# navigateur : http://localhost:5000  -> experiment "chab_appetence"
```

---

# Parcours B — Stack complète (MinIO + MLflow + Postgres + Airflow)

## B.1 Installer Docker

```bash
sudo apt install -y ca-certificates gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

## B.2 Démarrer l'infrastructure

```bash
docker compose up -d
docker compose ps
```

| Service | URL | Identifiants |
|---|---|---|
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| MLflow | http://localhost:5000 | — |
| Postgres | localhost:5432 | mlflow / mlflow |

## B.3 Basculer la configuration

```bash
sed -i 's|^STORAGE_BACKEND=.*|STORAGE_BACKEND=minio|' .env
sed -i 's|^MLFLOW_TRACKING_URI=.*|MLFLOW_TRACKING_URI=http://localhost:5000|' .env

export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

python -m src.config
make all
```

Contrôle des buckets :

```bash
docker run --rm --network host quay.io/minio/mc:latest /bin/sh -c \
  "mc alias set l http://localhost:9000 minioadmin minioadmin && mc ls -r l/chab-gold"
```

## B.4 Airflow

```bash
export AIRFLOW_HOME=~/airflow
export CHAB_PROJECT_HOME=~/chab-mlops
pip install "apache-airflow==2.9.2" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.2/constraints-3.10.txt"

airflow db migrate
airflow users create --username admin --password admin \
  --firstname Jihad --lastname Khalili --role Admin --email jihad@example.com

mkdir -p $AIRFLOW_HOME/dags
cp ~/chab-mlops/dags/*.py $AIRFLOW_HOME/dags/

airflow webserver -p 8080 -D
airflow scheduler -D
```

Déclencher :

```bash
airflow dags list | grep chab
airflow dags trigger chab_e2e_medallion_training
airflow dags trigger chab_batch_scoring
# UI : http://localhost:8080
```

---

## Étape finale — Fusionner dans main

```bash
cd ~/chab-mlops
git checkout main && git merge feature/multi-modeles && git push origin main
```

---

## Dépannage

| Symptôme | Cause | Correctif |
|---|---|---|
| `OSError: libgomp.so.1` | OpenMP absent | `sudo apt install -y libgomp1` |
| `NoCredentialsError` | variables MinIO non exportées | `export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...` |
| `ModuleNotFoundError: src` | lancé hors racine | exécuter depuis `~/chab-mlops` avec `python -m src.xxx` |
| MLflow `RESOURCE_DOES_NOT_EXIST` | modèle jamais enregistré | relancer `python -m src.train` avant `src.score` |
| CatBoost lent | trop d'itérations | `python -m src.train --fast` |
| `Port 5000 already in use` | MLflow déjà lancé | `fuser -k 5000/tcp` |
