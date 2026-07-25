"""Silver : nettoyage, typage, deduplication, normalisation -> Parquet."""
import hashlib
import pandas as pd
from src import config
from src.io_s3 import read_csv, write_parquet

RC_COLS = {
    "Identifiant Client": "client_id",
    "Business Line": "business_line",
    "Libelle Profil 1": "profil_1",
    "Libelle Profil 2": "profil_2",
    "Libelle Profil 3": "profil_3",
    "Libelle Court Rating Local": "rating",
    "Nom Localite (6eme Lgn) Fiscal": "ville",
    "Libelle CSP Local": "csp",
    "Libelle Segment": "segment",
    "Marche Segment": "marche",
    "Libelle Court Interdit": "interdit",
    "Code Sexe": "sexe",
    "Ind Type Pers. Morale ou Physique": "type_personne",
    "AGE_CLIENT": "age_client",
    "ANCIENNETE_MOIS": "anciennete_mois",
    "IS_MULTI_EQUIPE": "is_multi_equipe",
    "NB_PRODUITS_ACTIFS": "nb_produits_actifs",
    "NB_MOUVEMENTS_6M": "nb_mouvements_6m",
    "SNAPSHOT_ID": "snapshot_id",
}

PC_COLS = {
    "Identifiant Client": "client_id",
    "Code Type Contrat": "type_produit",
    "Solde Compte Engagement Devise Locale": "solde",
    "Encours Moy Credit": "encours_credit",
    "Encours Moy Debit": "encours_debit",
    "Libelle Categorie/Tarification": "pack",
    "Nombre Jours Compte Debiteur Gele": "jours_debiteur_gele",
    "TOP CAV": "top_cav", "TOP CEP": "top_cep", "TOP Cconso": "top_cconso",
    "TOP Package": "top_package", "TOP Carte": "top_carte",
    "TOP Client Impaye": "top_impaye",
    "CHAB": "chab", "IS_ACTIVE": "is_active",
    "AGE_PRODUIT": "age_produit", "SNAPSHOT_ID": "snapshot_id",
}

MVT_COLS = {
    "Identifiant SAB": "client_id",
    "Identifiant Contrat": "contrat_id",
    "Type Mouvement": "type_mvt",
    "Libelle Type Mouvement": "libelle_mvt",
    "Montant Devise Local": "montant",
    "Date Operation": "date_operation",
    "IS_CORRIGE": "is_corrige",
    "SOURCE_SYSTEME": "source_systeme",
    "SNAPSHOT_ID": "snapshot_id",
}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def clean_rc(snapshot_date: str):
    df = read_csv(config.BUCKET_BRONZE, f"snapshot_date={snapshot_date}/Vue_RC.csv")
    df = df[[c for c in RC_COLS if c in df.columns]].rename(columns=RC_COLS)
    # Normalisation casse (localites) et strings
    for c in ["ville", "segment", "profil_1", "profil_2", "profil_3", "csp", "marche"]:
        df[c] = df[c].astype(str).str.strip().str.upper().replace({"NAN": None})
    # Valeurs manquantes categorielles -> INCONNU
    df[["segment", "csp", "ville", "profil_2"]] = \
        df[["segment", "csp", "ville", "profil_2"]].fillna("INCONNU")
    # Interdit -> flag binaire
    df["flag_interdit"] = (df["interdit"].astype(str).str.upper()
                           .str.contains("INTERDIT", na=False)).astype(int)
    df = df.drop(columns=["interdit"])
    # Typage numerique + bornes de coherence
    for c in ["rating", "age_client", "anciennete_mois",
              "nb_produits_actifs", "nb_mouvements_6m"]:
        df[c] = _num(df[c])
    df.loc[~df["rating"].between(1, 10), "rating"] = None
    df.loc[~df["age_client"].between(0, 110), "age_client"] = None
    df.loc[df["anciennete_mois"] < 0, "anciennete_mois"] = None
    # Dedup securite sur la cle
    df = df.drop_duplicates(subset=["client_id"], keep="first")
    df["snapshot_date"] = snapshot_date
    write_parquet(df, config.BUCKET_SILVER, f"rc/snapshot_date={snapshot_date}/rc.parquet")


def clean_pc(snapshot_date: str):
    df = read_csv(config.BUCKET_BRONZE, f"snapshot_date={snapshot_date}/Vue_PC.csv")
    df = df[[c for c in PC_COLS if c in df.columns]].rename(columns=PC_COLS)
    df["type_produit"] = df["type_produit"].astype(str).str.strip().str.upper()
    df["pack"] = df["pack"].astype(str).str.strip().str.upper().replace({"NAN": "AUCUN"}).fillna("AUCUN")
    for c in ["solde", "encours_credit", "encours_debit",
              "jours_debiteur_gele", "age_produit"]:
        df[c] = _num(df[c]).fillna(0)
    for c in ["top_cav", "top_cep", "top_cconso", "top_package",
              "top_carte", "top_impaye", "chab", "is_active"]:
        df[c] = _num(df[c]).fillna(0).astype(int)
    # Winsorisation des soldes extremes (p1/p99) - traitement des outliers
    for c in ["solde", "encours_credit", "encours_debit"]:
        lo, hi = df[c].quantile([0.01, 0.99])
        df[c + "_w"] = df[c].clip(lo, hi)
    df = df.drop_duplicates()
    df["snapshot_date"] = snapshot_date
    write_parquet(df, config.BUCKET_SILVER, f"pc/snapshot_date={snapshot_date}/pc.parquet")


def clean_mvt(snapshot_date: str):
    df = read_csv(config.BUCKET_BRONZE, f"snapshot_date={snapshot_date}/Vue_MVT.csv")
    df = df[[c for c in MVT_COLS if c in df.columns]].rename(columns=MVT_COLS)
    n0 = len(df)
    # 1. Suppression des 4 800 doublons metier exacts
    df = df.drop_duplicates()
    print(f"[mvt] doublons supprimes: {n0 - len(df):,}")
    # 2. Normalisation des libelles en minuscules (~0,6 %)
    df["libelle_mvt"] = df["libelle_mvt"].astype(str).str.strip().str.upper()
    df["type_mvt"] = df["type_mvt"].astype(str).str.strip().str.upper()
    # 3. Typage
    df["montant"] = _num(df["montant"])
    df["date_operation"] = pd.to_datetime(df["date_operation"], errors="coerce")
    df["is_corrige"] = _num(df["is_corrige"]).fillna(0).astype(int)
    df = df.dropna(subset=["montant", "date_operation", "client_id"])
    # 4. Identifiant transactionnel deterministe (absent nativement)
    df["txn_id"] = (df["client_id"].astype(str) + "|" + df["type_mvt"] + "|"
                    + df["montant"].astype(str) + "|"
                    + df["date_operation"].astype(str) + "|"
                    + df.groupby(["client_id", "type_mvt", "montant",
                                  "date_operation"]).cumcount().astype(str)
                    ).map(lambda x: hashlib.md5(x.encode()).hexdigest())
    df["snapshot_date"] = snapshot_date
    write_parquet(df, config.BUCKET_SILVER, f"mvt/snapshot_date={snapshot_date}/mvt.parquet")


def run_all(snapshot_date: str):
    clean_rc(snapshot_date)
    clean_pc(snapshot_date)
    clean_mvt(snapshot_date)


if __name__ == "__main__":
    for d in config.SNAPSHOTS.values():
        run_all(d)
