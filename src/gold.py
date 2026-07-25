"""
Gold : feature engineering enrichi + cible (S1->S2) + population de scoring.

Améliorations vs v1 (pour dépasser AUC 0,74) :
  1. Signaux métier CHAB       : détention PEL / épargne, diversité produits, endettement conso.
  2. RFM comportemental        : récence, fréquence, montants, ratios débit/crédit.
  3. Salaire domicilié         : revenu mensuel estimé (capacité d'emprunt).
  4. Tendances temporelles     : pente du flux / des crédits sur la fenêtre S1,
                                 accélération de l'épargne (comportement de pré-achat).
  5. Ratios croisés            : solde/revenu, encours/revenu, taux d'épargne, charge conso.

Toutes les features sont calculées UNIQUEMENT sur les données observables au snapshot
(zero data leakage) : la cible est construite entre S1 et S2, jamais les features.
"""
import numpy as np
import pandas as pd

from src import config
from src.io_s3 import read_parquet, write_parquet

# Produits d'épargne pertinents pour l'appétence habitat (le PEL est le précurseur n°1)
PRODUITS_EPARGNE = ["CEP", "PEL", "DAT"]
# Mouvements entrants = revenus potentiels (VIR = virement reçu, VRST = versement espèces)
MVT_ENTRANTS = ["VIR", "VRST"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slope_by_group(df: pd.DataFrame, group: str, x: str, y: str) -> pd.Series:
    """Pente d'une régression linéaire y ~ x par groupe, vectorisée.
    slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2). Retourne 0 si variance nulle.
    """
    tmp = df[[group, x, y]].copy()
    tmp["xy"] = tmp[x] * tmp[y]
    tmp["xx"] = tmp[x] * tmp[x]
    g = tmp.groupby(group)
    n = g[x].count()
    sx, sy = g[x].sum(), g[y].sum()
    sxy, sxx = g["xy"].sum(), g["xx"].sum()
    denom = (n * sxx - sx ** 2)
    slope = (n * sxy - sx * sy) / denom.replace(0, np.nan)
    return slope.fillna(0.0)


# ---------------------------------------------------------------------------
# RC : profil client
# ---------------------------------------------------------------------------
def features_rc(snapshot_date: str) -> pd.DataFrame:
    rc = read_parquet(config.BUCKET_SILVER, f"rc/snapshot_date={snapshot_date}/rc.parquet")

    # Fenêtre d'âge d'achat immobilier (28-45 ans = coeur de cible habitat)
    rc["age_prime_immo"] = rc["age_client"].between(28, 45).astype(int)
    rc["age_client"] = rc["age_client"].fillna(rc["age_client"].median())

    # Profil de stabilité (Profil 3 : TRES STABLE / STABLE / VOLATIL / A RISQUE)
    prof3 = rc.get("profil_3", pd.Series(index=rc.index, dtype=object)).astype(str).str.upper()
    rc["profil_stable"] = prof3.isin(["TRES STABLE", "STABLE"]).astype(int)

    # Segment à revenu latent élevé
    seg = rc.get("segment", pd.Series(index=rc.index, dtype=object)).astype(str).str.upper()
    rc["segment_aise"] = seg.str.contains("AFFLUENT|PREMIUM|POTENTIEL", na=False).astype(int)

    rc["anciennete_mois"] = rc["anciennete_mois"].fillna(0)
    rc["anciennete_annees"] = rc["anciennete_mois"] / 12.0

    keep = ["client_id", "business_line", "profil_1", "profil_2", "profil_3",
            "segment", "marche", "csp", "ville", "sexe", "type_personne",
            "rating", "age_client", "age_prime_immo", "anciennete_mois",
            "anciennete_annees", "profil_stable", "segment_aise",
            "flag_interdit", "is_multi_equipe", "nb_produits_actifs",
            "nb_mouvements_6m"]
    return rc[[c for c in keep if c in rc.columns]]


# ---------------------------------------------------------------------------
# PC : produits détenus (+ signaux épargne / habitat)
# ---------------------------------------------------------------------------
def features_pc(snapshot_date: str) -> pd.DataFrame:
    pc = read_parquet(config.BUCKET_SILVER, f"pc/snapshot_date={snapshot_date}/pc.parquet")
    actif = pc[pc["is_active"] == 1].copy()

    # Nb de produits actifs par type (pivot) -> nb_CAV, nb_CEP, nb_PEL, ...
    piv = (actif.pivot_table(index="client_id", columns="type_produit",
                             values="is_active", aggfunc="sum", fill_value=0)
           .add_prefix("nb_").reset_index())

    agg = actif.groupby("client_id").agg(
        nb_produits=("type_produit", "count"),
        nb_produits_distincts=("type_produit", "nunique"),
        solde_total=("solde_w", "sum"),
        encours_credit_total=("encours_credit_w", "sum"),
        encours_debit_total=("encours_debit_w", "sum"),
        anciennete_produit_max=("age_produit", "max"),
        anciennete_produit_moy=("age_produit", "mean"),
        jours_debiteur_max=("jours_debiteur_gele", "max"),
        top_carte=("top_carte", "max"),
        top_package=("top_package", "max"),
        top_cconso=("top_cconso", "max"),
        top_impaye=("top_impaye", "max"),
        has_chab=("chab", "max"),
    ).reset_index()

    # Pack principal
    packs = (actif[actif["pack"] != "AUCUN"]
             .groupby("client_id")["pack"].first().rename("pack").reset_index())

    out = agg.merge(piv, on="client_id", how="left").merge(packs, on="client_id", how="left")
    out["pack"] = out["pack"].fillna("AUCUN")

    # ----- Signaux épargne / habitat (fort pouvoir prédictif CHAB) -----
    for p in PRODUITS_EPARGNE + ["CCONSO"]:
        col = f"nb_{p}"
        if col not in out.columns:
            out[col] = 0
    out["detient_pel"] = (out["nb_PEL"] > 0).astype(int)
    out["detient_epargne"] = (out[[f"nb_{p}" for p in PRODUITS_EPARGNE]].sum(axis=1) > 0).astype(int)
    out["nb_produits_epargne"] = out[[f"nb_{p}" for p in PRODUITS_EPARGNE]].sum(axis=1)
    out["detient_cconso"] = (out["nb_CCONSO"] > 0).astype(int)
    # Diversité = nb de types distincts détenus / nb de types possibles
    out["diversite_produits"] = out["nb_produits_distincts"] / 8.0

    return out


# ---------------------------------------------------------------------------
# MVT : RFM + salaire + tendances temporelles
# ---------------------------------------------------------------------------
def features_mvt(snapshot_date: str) -> pd.DataFrame:
    mvt = read_parquet(config.BUCKET_SILVER, f"mvt/snapshot_date={snapshot_date}/mvt.parquet")
    ref = pd.Timestamp(snapshot_date)
    mvt["credit"] = mvt["montant"].clip(lower=0)
    mvt["debit"] = (-mvt["montant"]).clip(lower=0)
    mvt["mois"] = mvt["date_operation"].dt.to_period("M")

    # ----- Agrégats RFM globaux -----
    agg = mvt.groupby("client_id").agg(
        nb_mvt=("txn_id", "count"),
        total_credits=("credit", "sum"),
        total_debits=("debit", "sum"),
        montant_moyen=("montant", "mean"),
        montant_std=("montant", "std"),
        derniere_operation=("date_operation", "max"),
        nb_mois_actifs=("date_operation", lambda s: s.dt.to_period("M").nunique()),
    ).reset_index()
    agg["recence_jours"] = (ref - agg["derniere_operation"]).dt.days
    agg["flux_net"] = agg["total_credits"] - agg["total_debits"]
    agg["ratio_debit_credit"] = agg["total_debits"] / (agg["total_credits"] + 1)
    agg["taux_epargne"] = agg["flux_net"] / (agg["total_credits"] + 1)
    agg["montant_std"] = agg["montant_std"].fillna(0)
    agg = agg.drop(columns=["derniere_operation"])

    # Nb de mouvements par type (VIR, PRLV, CB, ...)
    piv = (mvt.pivot_table(index="client_id", columns="type_mvt",
                           values="txn_id", aggfunc="count", fill_value=0)
           .add_prefix("nb_mvt_").reset_index())

    # ----- Salaire domicilié (revenu récurrent stable) -----
    entrants = mvt[mvt["type_mvt"].isin(MVT_ENTRANTS) & (mvt["montant"] > 0)]
    sal_mens = (entrants.groupby(["client_id", "mois"])
                .agg(montant_max=("montant", "max")).reset_index())
    sal = sal_mens.groupby("client_id").agg(
        nb_mois_revenu=("mois", "nunique"),
        revenu_mensuel_estime=("montant_max", "median"),
        revenu_stabilite=("montant_max", lambda s: s.std() / (s.mean() + 1)),
    ).reset_index()
    sal["revenu_stabilite"] = sal["revenu_stabilite"].fillna(0)
    sal["salaire_domicilie"] = (
        (sal["nb_mois_revenu"] >= 4) & (sal["revenu_stabilite"] < 0.35)
    ).astype(int)

    # ----- Tendances temporelles sur la fenêtre (comportement de pré-achat) -----
    mens = (mvt.groupby(["client_id", "mois"])
            .agg(flux=("montant", "sum"), credits=("credit", "sum"))
            .reset_index())
    mens["rang"] = mens.groupby("client_id")["mois"].rank(method="dense")
    pente_flux = _slope_by_group(mens, "client_id", "rang", "flux").rename("tendance_flux")
    pente_cred = _slope_by_group(mens, "client_id", "rang", "credits").rename("tendance_credits")

    # Accélération épargne = flux des 3 derniers mois - flux des 3 premiers mois
    rmax = mens.groupby("client_id")["rang"].transform("max")
    recent = mens[mens["rang"] > rmax - 3].groupby("client_id")["flux"].sum().rename("flux_recent")
    ancien = mens[mens["rang"] <= 3].groupby("client_id")["flux"].sum().rename("flux_ancien")
    accel = (recent.to_frame().join(ancien, how="outer").fillna(0))
    accel["acceleration_epargne"] = accel["flux_recent"] - accel["flux_ancien"]

    trends = (pente_flux.to_frame().join([pente_cred, accel["acceleration_epargne"]], how="outer")
              .reset_index())

    return (agg.merge(piv, on="client_id", how="left")
               .merge(sal, on="client_id", how="left")
               .merge(trends, on="client_id", how="left"))


# ---------------------------------------------------------------------------
# Assemblage des features
# ---------------------------------------------------------------------------
def build_features(snapshot_date: str) -> pd.DataFrame:
    rc = features_rc(snapshot_date)
    feats = (rc.merge(features_pc(snapshot_date), on="client_id", how="left")
               .merge(features_mvt(snapshot_date), on="client_id", how="left"))

    # Compteurs / montants manquants -> 0 (absence logique d'événement)
    num_zero = [c for c in feats.columns
                if c.startswith(("nb_", "total_", "top_", "tendance_", "flux"))
                or c in ("has_chab", "detient_pel", "detient_epargne", "detient_cconso",
                         "nb_produits_epargne", "diversite_produits", "solde_total",
                         "encours_credit_total", "encours_debit_total", "flux_net",
                         "acceleration_epargne", "salaire_domicilie",
                         "revenu_mensuel_estime", "nb_mois_revenu", "taux_epargne",
                         "ratio_debit_credit", "nb_mois_actifs", "nb_mvt")]
    feats[num_zero] = feats[num_zero].fillna(0)
    feats["pack"] = feats["pack"].fillna("AUCUN")
    feats["recence_jours"] = feats["recence_jours"].fillna(999)
    feats["revenu_mensuel_estime"] = feats["revenu_mensuel_estime"].fillna(0)

    # ----- Ratios croisés (normalisent par la capacité financière) -----
    revenu = feats["revenu_mensuel_estime"].replace(0, np.nan)
    feats["ratio_solde_revenu"] = (feats["solde_total"] / revenu).fillna(0)
    feats["charge_conso_revenu"] = (feats["encours_debit_total"] / revenu).fillna(0)
    feats["epargne_par_produit"] = feats["solde_total"] / (feats["nb_produits"].replace(0, np.nan))
    feats["epargne_par_produit"] = feats["epargne_par_produit"].fillna(0)
    feats["mvt_par_produit"] = feats["nb_mvt"] / (feats["nb_produits"].replace(0, np.nan))
    feats["mvt_par_produit"] = feats["mvt_par_produit"].fillna(0)

    # Nettoyage des inf issus des divisions
    feats = feats.replace([np.inf, -np.inf], 0)

    write_parquet(feats, config.BUCKET_GOLD,
                  f"features/snapshot_date={snapshot_date}/features.parquet")
    return feats


# ---------------------------------------------------------------------------
# Cible (S1 -> S2) et population de scoring
# ---------------------------------------------------------------------------
def build_training_set():
    s1, s2 = config.SNAPSHOTS["S1"], config.SNAPSHOTS["S2"]
    f1 = read_parquet(config.BUCKET_GOLD, f"features/snapshot_date={s1}/features.parquet")
    pc2 = read_parquet(config.BUCKET_SILVER, f"pc/snapshot_date={s2}/pc.parquet")

    chab_s2 = (pc2[(pc2["chab"] == 1) & (pc2["is_active"] == 1)]
               .groupby("client_id").size().rename("chab_s2").reset_index())

    # Population = clients SANS CHAB a S1 ; label = acquisition a S2
    pop = f1[f1["has_chab"] == 0].copy()
    pop = pop.merge(chab_s2, on="client_id", how="left")
    pop["target"] = (pop["chab_s2"].fillna(0) > 0).astype(int)
    pop = pop.drop(columns=["chab_s2", "has_chab"])
    print(f"[gold] population={len(pop):,} | positifs={pop['target'].sum():,} "
          f"({pop['target'].mean():.2%})")
    write_parquet(pop, config.BUCKET_GOLD, "training/chab_training_set.parquet")


def build_scoring_set():
    s2 = config.SNAPSHOTS["S2"]
    f2 = read_parquet(config.BUCKET_GOLD, f"features/snapshot_date={s2}/features.parquet")
    pop = f2[f2["has_chab"] == 0].drop(columns=["has_chab"])
    write_parquet(pop, config.BUCKET_GOLD,
                  f"scoring/snapshot_date={s2}/scoring_population.parquet")


if __name__ == "__main__":
    for d in config.SNAPSHOTS.values():
        build_features(d)
    build_training_set()
    build_scoring_set()