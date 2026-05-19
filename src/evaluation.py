"""Protocollo di valutazione del sistema di anomaly detection.

In assenza di etichette di ground truth, la traccia (sez. Fase 3.b) impone
di costruire un protocollo di valutazione esplicito combinando più strategie.
Questo modulo implementa tutte le metriche e le analisi necessarie.

Tre strategie combinate (vedi PIANO.md sez. 4 Fase 3b)
-------------------------------------------------------
1. **Anomalie sintetiche**: confronto alert/score vs ground truth proxy
   iniettato → recall stratificato per tipologia, precision, F-beta, AUC-PR.

2. **Eventi reali**: lista manuale di eventi storici noti → hit rate.

3. **Stabilità**: variazione degli iperparametri → Jaccard tra insiemi di alert.

Metriche custom motivate dal framing
------------------------------------
- **F-0.5 score** (β=0.5): pesa la precisione 2x rispetto al recall.
  Coerente col framing: utente analista, costo FP > costo FN.
- **Operational alert rate**: percentuale ore-spira con alert generato.
  Deve restare ≤ 0.5% per essere leggibile dall'analista (~ qualche decina
  di alert/settimana su 18 spire).
- **AUC-PR**: area sotto la curva precision-recall. Insensibile al class
  imbalance (le anomalie sono per definizione rare). Preferita ad AUC-ROC
  per task fortemente sbilanciati come questo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


# =============================================================================
# Strategia 1 — Anomalie sintetiche
# =============================================================================
def valuta_su_anomalie_sintetiche(
    df_scores: pd.DataFrame,
    df_ground_truth: pd.DataFrame,
    col_score: str = "score_ensemble",
    col_alert: str = "alert_ensemble",
    beta: float = 0.5,
) -> dict:
    """Calcola metriche del sistema rispetto alle anomalie sintetiche iniettate.

    Parametri
    ---------
    df_scores : DataFrame con colonne (chiave, timestamp, score, alert)
                — output del sistema/ensemble.
    df_ground_truth : DataFrame con colonne (chiave, timestamp, tipo, ...)
                      — generato da `synthetic_anomalies.anomalie_a_dataframe`.
    col_score, col_alert : colonne da valutare
    beta : peso del recall rispetto alla precision (β<1 → precision conta di più)

    Ritorna
    -------
    dict con:
      - precision, recall, f_beta
      - auc_pr (area under PR curve)
      - alert_rate (operational)
      - recall_per_tipo (dict)
      - n_anomalie_gt, n_alert
    """
    # Etichetta ogni riga del dataset scored come "vera anomalia" se
    # corrispondente nel ground truth.
    df = df_scores.merge(
        df_ground_truth[["chiave", "timestamp", "tipo"]],
        on=["chiave", "timestamp"],
        how="left",
    )
    df["y_true"] = df["tipo"].notna().astype(int)
    df["y_pred"] = df[col_alert].astype(int)

    # Metriche standard.
    prec = precision_score(df["y_true"], df["y_pred"], zero_division=0)
    rec = recall_score(df["y_true"], df["y_pred"], zero_division=0)
    fbeta = fbeta_score(df["y_true"], df["y_pred"], beta=beta, zero_division=0)
    auc_pr = (
        average_precision_score(df["y_true"], df[col_score])
        if df["y_true"].sum() > 0 else float("nan")
    )
    alert_rate = df["y_pred"].mean()

    # Recall stratificato per tipo di anomalia.
    recall_per_tipo = {}
    for tipo, sub in df[df["y_true"] == 1].groupby("tipo"):
        recall_per_tipo[tipo] = float(sub["y_pred"].mean())

    return {
        "precision": float(prec),
        "recall": float(rec),
        f"f{beta}": float(fbeta),
        "auc_pr": float(auc_pr),
        "alert_rate": float(alert_rate),
        "recall_per_tipo": recall_per_tipo,
        "n_anomalie_gt": int(df["y_true"].sum()),
        "n_alert_predetti": int(df["y_pred"].sum()),
    }


# =============================================================================
# Strategia 2 — Eventi reali
# =============================================================================
@dataclass
class EventoStoricoNoto:
    """Descrittore di un evento cittadino reale (validazione qualitativa)."""
    nome: str
    data_inizio: pd.Timestamp
    data_fine: pd.Timestamp
    impatto_atteso: str  # "drop" / "spike" / "shift"
    spire_attese: list[int] | None = None  # None = qualsiasi
    fonte: str = ""


def valuta_su_eventi_reali(
    df_scores: pd.DataFrame,
    eventi: list[EventoStoricoNoto],
    col_alert: str = "alert_ensemble",
    finestra_ore_match: int = 6,
) -> pd.DataFrame:
    """Per ogni evento storico, verifica se il sistema ha prodotto alert
    nella finestra temporale (espansa di ±finestra_ore_match per gestire
    lieve incertezza nei timestamp dell'evento).

    Ritorna un DataFrame con una riga per evento e colonne:
      - n_alert_in_finestra
      - colpito (bool: almeno 1 alert su spira ammessa nella finestra)
      - spire_che_hanno_segnalato
    """
    righe = []
    for ev in eventi:
        start = ev.data_inizio - pd.Timedelta(hours=finestra_ore_match)
        end = ev.data_fine + pd.Timedelta(hours=finestra_ore_match)

        mask = (df_scores["timestamp"] >= start) & (df_scores["timestamp"] <= end)
        if ev.spire_attese is not None:
            mask &= df_scores["chiave"].isin(ev.spire_attese)

        sotto = df_scores[mask & df_scores[col_alert]]
        righe.append({
            "evento": ev.nome,
            "data": ev.data_inizio.date(),
            "impatto_atteso": ev.impatto_atteso,
            "n_alert_in_finestra": int(len(sotto)),
            "colpito": bool(len(sotto) > 0),
            "spire_che_hanno_segnalato": sotto["chiave"].unique().tolist(),
        })
    return pd.DataFrame(righe)


def eventi_default_bologna() -> list[EventoStoricoNoto]:
    """Lista di eventi cittadini reali del 2024-2025 per la validazione.

    NB: lista compilata "best-effort" da fonti aperte (Repubblica Bologna,
    BolognaToday, archivio comunale). In un progetto reale andrebbe
    confrontata con i registri ufficiali del settore Mobilità.
    """
    return [
        EventoStoricoNoto(
            nome="Sciopero TPL nazionale",
            data_inizio=pd.Timestamp("2024-04-12 09:00"),
            data_fine=pd.Timestamp("2024-04-12 17:00"),
            impatto_atteso="shift",
            fonte="Repubblica Bologna 2024-04-11",
        ),
        EventoStoricoNoto(
            nome="Festa della Repubblica (parata)",
            data_inizio=pd.Timestamp("2024-06-02 09:00"),
            data_fine=pd.Timestamp("2024-06-02 13:00"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Bologna FC home match vs Juventus",
            data_inizio=pd.Timestamp("2024-05-20 19:00"),
            data_fine=pd.Timestamp("2024-05-20 23:00"),
            impatto_atteso="spike",
            fonte="Calendario Serie A 2023-24",
        ),
        EventoStoricoNoto(
            nome="Ferragosto 2024",
            data_inizio=pd.Timestamp("2024-08-15 00:00"),
            data_fine=pd.Timestamp("2024-08-15 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Black Friday 2024 (concentramenti commerciali)",
            data_inizio=pd.Timestamp("2024-11-29 14:00"),
            data_fine=pd.Timestamp("2024-11-29 21:00"),
            impatto_atteso="spike",
            fonte="Cronaca commerciale",
        ),
        EventoStoricoNoto(
            nome="Capodanno 2025",
            data_inizio=pd.Timestamp("2025-01-01 00:00"),
            data_fine=pd.Timestamp("2025-01-01 06:00"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Pasqua 2024",
            data_inizio=pd.Timestamp("2024-03-31 00:00"),
            data_fine=pd.Timestamp("2024-03-31 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Festa San Petronio 2024",
            data_inizio=pd.Timestamp("2024-10-04 00:00"),
            data_fine=pd.Timestamp("2024-10-04 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile bolognese",
        ),
        EventoStoricoNoto(
            nome="Sciopero generale 2024-10-18",
            data_inizio=pd.Timestamp("2024-10-18 09:00"),
            data_fine=pd.Timestamp("2024-10-18 17:00"),
            impatto_atteso="shift",
            fonte="CGIL nazionale",
        ),
        EventoStoricoNoto(
            nome="25 Aprile 2024 (Liberazione)",
            data_inizio=pd.Timestamp("2024-04-25 00:00"),
            data_fine=pd.Timestamp("2024-04-25 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="1 Maggio 2024",
            data_inizio=pd.Timestamp("2024-05-01 00:00"),
            data_fine=pd.Timestamp("2024-05-01 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Inaugurazione anno accademico UniBO 2024",
            data_inizio=pd.Timestamp("2024-09-30 09:00"),
            data_fine=pd.Timestamp("2024-09-30 13:00"),
            impatto_atteso="spike",
            fonte="Calendario UniBO",
        ),
        EventoStoricoNoto(
            nome="Mercato Antiquario Piazza S.Stefano (1° dom mese)",
            data_inizio=pd.Timestamp("2024-09-01 08:00"),
            data_fine=pd.Timestamp("2024-09-01 19:00"),
            impatto_atteso="shift",
            fonte="Eventi cittadini ricorrenti",
        ),
        EventoStoricoNoto(
            nome="Immacolata 2024",
            data_inizio=pd.Timestamp("2024-12-08 00:00"),
            data_fine=pd.Timestamp("2024-12-08 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Natale 2024",
            data_inizio=pd.Timestamp("2024-12-25 00:00"),
            data_fine=pd.Timestamp("2024-12-25 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Pasqua 2025",
            data_inizio=pd.Timestamp("2025-04-20 00:00"),
            data_fine=pd.Timestamp("2025-04-20 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Festa della Repubblica 2025",
            data_inizio=pd.Timestamp("2025-06-02 00:00"),
            data_fine=pd.Timestamp("2025-06-02 23:59"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
        EventoStoricoNoto(
            nome="Capodanno 2025-2026 (notte di San Silvestro)",
            data_inizio=pd.Timestamp("2025-12-31 22:00"),
            data_fine=pd.Timestamp("2026-01-01 02:00"),
            impatto_atteso="drop",
            fonte="Calendario civile",
        ),
    ]


# =============================================================================
# Strategia 3 — Stabilità degli alert al variare degli iperparametri
# =============================================================================
def jaccard_alert_sets(set_a: pd.DataFrame, set_b: pd.DataFrame) -> float:
    """Calcola il Jaccard index tra due insiemi di alert.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Range: [0, 1]. 1 = stessi alert. 0 = disjoint.

    Interpretazione (vedi PIANO.md sez. 4 Fase 3b)
    -----------------------------------------------
    Un sistema stabile produce un "core" di alert robusto rispetto a piccole
    variazioni di iperparametri. Se la J(α₁, α₂) crolla per piccoli cambi,
    significa che il sistema è fragile e i suoi alert non sono affidabili.
    """
    a = set(map(tuple, set_a[["chiave", "timestamp"]].values))
    b = set(map(tuple, set_b[["chiave", "timestamp"]].values))
    if not a and not b:
        return 1.0
    if not (a | b):
        return 0.0
    return len(a & b) / len(a | b)


def analisi_stabilita(
    df_master: pd.DataFrame,
    funzione_ensemble,
    funzione_baseline_1,
    funzione_baseline_2,
    funzione_baseline_3,
    percentili: list[float] = [95, 97, 99, 99.5, 99.9],
) -> pd.DataFrame:
    """Genera alert al variare del percentile di soglia ensemble e ne
    misura la stabilità reciproca tramite matrice di Jaccard.

    Output: DataFrame quadrato con percentili in righe/colonne e J come valore.
    """
    df_b1 = funzione_baseline_1(df_master)
    df_b2 = funzione_baseline_2(df_master)
    df_b3 = funzione_baseline_3(df_master)

    alert_sets: dict[float, pd.DataFrame] = {}
    for p in percentili:
        ens = funzione_ensemble(df_b1, df_b2, df_b3, percentile_soglia=p)
        alert_sets[p] = ens[ens["alert_ensemble"]][["chiave", "timestamp"]]

    matrice = pd.DataFrame(index=percentili, columns=percentili, dtype=float)
    for p1 in percentili:
        for p2 in percentili:
            matrice.at[p1, p2] = jaccard_alert_sets(alert_sets[p1], alert_sets[p2])

    return matrice


# =============================================================================
# Curva precision-recall + soglia ottimale
# =============================================================================
def curva_pr_e_soglia_ottimale(
    df_scores: pd.DataFrame,
    df_ground_truth: pd.DataFrame,
    col_score: str = "score_ensemble",
    beta: float = 0.5,
) -> tuple[pd.DataFrame, float]:
    """Calcola la curva PR e trova la soglia che massimizza F-beta.

    Utile per il report: mostra il trade-off complessivo precision/recall
    e identifica la "soglia ideale" (post-hoc) coerente con il framing.
    """
    df = df_scores.merge(
        df_ground_truth[["chiave", "timestamp", "tipo"]],
        on=["chiave", "timestamp"], how="left",
    )
    df["y_true"] = df["tipo"].notna().astype(int)

    if df["y_true"].sum() == 0:
        return pd.DataFrame(), float("nan")

    prec, rec, thr = precision_recall_curve(df["y_true"], df[col_score])
    # F-beta su ciascun (prec, rec). prec e rec hanno len(thr)+1, l'ultimo
    # è il punto degenere prec=1, rec=0.
    fbeta = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
    idx_best = int(np.argmax(fbeta[:-1])) if len(thr) > 0 else 0
    soglia_ott = float(thr[idx_best]) if len(thr) > 0 else float("nan")

    curva = pd.DataFrame({
        "soglia": np.concatenate([thr, [np.inf]]),
        "precision": prec,
        "recall": rec,
        f"f{beta}": fbeta,
    })
    return curva, soglia_ott
