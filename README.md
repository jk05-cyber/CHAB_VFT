# CHAB_VFT — Score d'appétence Crédit Habitat (pipeline MLOps end-to-end)

Pipeline **MLOps de bout en bout** qui transforme des extractions bancaires brutes (CSV) en un
**score d'appétence au Crédit Habitat (CHAB)** par client :
architecture **Médaillon (Bronze / Silver / Gold)**, **analyses statistiques complètes**,
**benchmark multi-modèles** (RL, RF, XGBoost, LightGBM, CatBoost), tracking **MLflow** et
orchestration **Airflow**.

> Démarche **CRISP-DM** : compréhension métier → compréhension des données → préparation →
> modélisation → évaluation → déploiement.

👉 **Mise en place pas-à-pas sous Ubuntu : voir [`WALKTHROUGH.md`](WALKTHROUGH.md)**

---

## 1. Objectif métier

Identifier, parmi les clients **ne détenant pas encore de crédit habitat**, ceux qui ont la plus
forte probabilité d'en souscrire, afin de concentrer les campagnes sur les déciles à fort potentiel.

| Snapshot | Date | Rôle |
|---|---|---|
| `S1` | 2026-01-01 | construction des **features** |
| `S2` | 2026-06-01 | observation de la **cible** + population à scorer |

**Cible** : population = clients sans CHAB actif en S1 ; `target = 1` si CHAB actif en S2.
**Anti data-leakage** : aucune information postérieure à S1 n'entre dans les features ;
l'imputation (médianes) et la winsorisation sont calculées sur le train uniquement.

---

## 2. Architecture

```
  CSV bruts            BRONZE              SILVER               GOLD
  Vue_RC.csv    ──►  copie immuable  ──►  nettoyage      ──►  features (S1/S2)
  Vue_PC.csv         partition par        typage/dédup         training_set (+ target)
  Vue_MVT.csv        snapshot_date        winsorisation        scoring_population
                                          txn_id MD5                  │
                                                        ┌────────────┼────────────┐
                                                        ▼                         ▼
                                               Analyses statistiques      Benchmark 5 modèles
                                               khi2 / Cramér V / IV       RL · RF · XGB · LGBM · CatBoost
                                               Mann-Whitney / KS          → MLflow Registry
                                               VIF / PSI / forward                │
                                                                                  ▼
                                                                        Scoring batch + déciles
```

Stockage : **MinIO (S3)** en production, **système de fichiers local** en développement
(bascule par la variable `STORAGE_BACKEND`).

---

## 3. Structure du dépôt

```
chab-mlops/
├── dags/
│   ├── chab_e2e_pipeline.py      # bronze → silver → gold → stats → benchmark
│   └── chab_scoring.py           # scoring batch mensuel
├── scripts/
│   └── generate_synthetic_landing.py  # jeu de données de démo réaliste
├── src/
│   ├── config.py         # variables d'env, snapshots, backend de stockage
│   ├── io_s3.py          # I/O MinIO ou local (même API)
│   ├── bronze.py         # ingestion brute
│   ├── silver.py         # nettoyage / typage / dédup / winsorisation
│   ├── gold.py           # feature engineering + cible + population de scoring
│   ├── stats_analysis.py # khi2, Cramér's V, Mann-Whitney, KS, IV, VIF, PSI, forward selection
│   ├── models.py         # zoo de modèles + preprocessing partagé + grilles d'hyperparamètres
│   ├── metrics.py        # toutes les métriques + tests statistiques + graphiques
│   ├── train.py          # benchmark multi-modèles + MLflow + champion
│   └── score.py          # scoring batch + déciles + segments d'action
├── tests/test_pipeline.py
├── docker-compose.yml    # MinIO + Postgres + MLflow
├── Makefile              # make all
├── requirements.txt
├── .env.example
├── README.md
└── WALKTHROUGH.md        # guide Ubuntu pas-à-pas
```

---

## 4. Démarrage rapide

```bash
git clone https://github.com/jk05-cyber/CHAB_VFT.git chab-mlops && cd chab-mlops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
make all        # data → bronze → silver → gold → stats → train → score
```

---

## 5. Couches du pipeline

### 5.1 Silver — nettoyage
Renommage vers un schéma technique, normalisation des chaînes, `INCONNU` pour les catégoriels
manquants, flags binaires (`flag_interdit`), bornes de cohérence (rating 1-10, âge 0-110),
déduplication, **winsorisation P1/P99** des montants, **`txn_id` MD5 déterministe**.

### 5.2 Gold — feature engineering

| Famille | Exemples |
|---|---|
| Profil client | `age_prime_immo` (28-45 ans), `profil_stable`, `segment_aise`, `anciennete_annees`, `rating` |
| Détention produits | `nb_produits`, `detient_pel`, `detient_epargne`, `detient_cconso`, `diversite_produits` |
| RFM transactionnel | `recence_jours`, `nb_mvt`, `nb_mois_actifs`, `flux_net`, `ratio_debit_credit`, `taux_epargne` |
| Revenu | `revenu_mensuel_estime`, `revenu_stabilite`, `salaire_domicilie` |
| Tendances | `tendance_flux`, `tendance_credits`, `acceleration_epargne` |
| Ratios croisés | `ratio_solde_revenu`, `charge_conso_revenu`, `epargne_par_produit`, `mvt_par_produit` |

Signal métier clé : la **détention d'un PEL** est le précurseur n° 1 de l'appétence habitat.

---

## 6. Analyses statistiques (`python -m src.stats_analysis`)

| Analyse | Test / indicateur | Fichier produit |
|---|---|---|
| Profil du dataset | volumétrie, taux de positifs, ratio de déséquilibre | `profil_dataset.json` |
| Qualité | taux de valeurs manquantes | `missing_rates.csv` |
| Numérique vs cible | **Mann-Whitney U**, **point-biserial r**, **Kolmogorov-Smirnov**, **Cohen's d**, **IV** | `stats_numeriques.csv` |
| Catégoriel vs cible | **Khi-deux**, ddl, **Cramér's V** (corrigé Bergsma), **IV/WOE** | `stats_categorielles.csv` |
| Multicolinéarité | corrélation de **Spearman**, **VIF** | `correlation_spearman.csv`, `vif.csv` |
| Stabilité / drift | **PSI** S1 vs S2 (seuils 0,10 / 0,25) | `psi_s1_vs_s2.csv` |
| Sélection | **Forward selection** guidée par l'AUC | `forward_selection.csv` |

---

## 7. Benchmark multi-modèles (`python -m src.train`)

Tous les modèles partagent **le même preprocessing** (imputation médiane + One-Hot avec
`min_frequency=50`), afin que la comparaison soit strictement équitable.

| Modèle | Rôle | Particularités |
|---|---|---|
| `dummy` | borne du hasard | stratégie `prior` |
| `logistic_regression` | **baseline linéaire (RL)** | standardisation + `class_weight=balanced` |
| `random_forest` | **baseline ensembliste (RF)** | `balanced_subsample`, `max_depth=14` |
| `xgboost` | boosting | `hist`, `eval_metric=aucpr`, `scale_pos_weight` |
| `lightgbm` | boosting | `num_leaves=63`, `subsample=0.8` |
| `catboost` | boosting | `eval_metric=PRAUC`, `depth=6` |

Options : `--fast`, `--models a,b`, `--tune` (RandomizedSearchCV), `--no-cv`,
`--no-calibration`, `--sample N`.

### Métriques calculées (toutes, pour chaque modèle)

| Famille | Métriques |
|---|---|
| Discrimination | **ROC-AUC**, **Gini**, **PR-AUC**, **KS** |
| Calibration | **LogLoss**, **Brier**, courbe de calibration |
| Métier | **Lift @ déciles 1/2/3**, **capture top 10/20/30 %**, **precision@10 %** |
| À seuil optimal (F1) | accuracy, balanced accuracy, precision, **recall**, spécificité, NPV, **F1**, **F2**, **MCC**, kappa, matrice de confusion (TP/FP/FN/TN) |
| Robustesse | PR-AUC en **validation croisée** (moyenne ± écart-type), temps d'entraînement |
| Tests statistiques | **IC 95 % bootstrap de l'AUC**, **test bootstrap apparié** champion vs challengers, **test de McNemar** |

### Graphiques générés (`artifacts/models/figures/`)
`roc_curves.png` · `pr_curves.png` · `gain_curve.png` · `lift_by_decile.png` ·
`calibration.png` · `score_distribution.png`

Le **champion** est sélectionné sur `CHAMPION_METRIC` (par défaut PR-AUC, pertinent en contexte
déséquilibré), **calibré** (isotonique) puis enregistré dans le **MLflow Model Registry**.

---

## 8. Scoring batch (`python -m src.score`)

| Colonne | Description |
|---|---|
| `client_id` | identifiant client |
| `score_appetence_chab` | probabilité calibrée (0-1) |
| `decile` | 10 (meilleur) → 1 |
| `segment_action` | `Prioritaire` (D10) · `A travailler` (D8-9) · `Veille` |

Sorties : `gold/scores/snapshot_date=.../scores_chab.parquet` +
`artifacts/scoring/top_clients_<date>.csv`.

---

## 9. Orchestration

| DAG | Enchaînement | Planification |
|---|---|---|
| `chab_e2e_medallion_training` | bronze → silver (RC/PC/MVT × S1/S2) → gold → stats → benchmark | manuel (`@monthly` en prod) |
| `chab_batch_scoring` | scoring de la population S2 | manuel (`@monthly` en prod) |

Le chemin du projet est lu depuis la variable d'environnement `CHAB_PROJECT_HOME`.

---

## 10. Perspectives

- Backend MLflow distant (Postgres + artefacts S3) déjà fourni dans `docker-compose.yml`
- Explicabilité SHAP par client (assistant conseiller)
- Monitoring de drift automatisé (PSI en tâche planifiée + alerting)
- Validation de données (Great Expectations / Pandera) et CI/CD GitHub Actions

---

## Auteur

**Khalili Jihad** — [@jk05-cyber](https://github.com/jk05-cyber) — Casablanca
