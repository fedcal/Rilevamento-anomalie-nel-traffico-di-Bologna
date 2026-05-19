"""Tre baseline di anomaly detection + sistema ensemble.

Implementazione delle tre famiglie di approccio richieste dalla traccia
(sez. 2.b "Costruzione di tre baseline"):

1. **Baseline 1 — Statistica classica**: STL decomposition + soglia su residui
   standardizzati via MAD (Median Absolute Deviation).

2. **Baseline 2 — Distance/density-based**: IsolationForest su feature
   engineered (lag, calendario, meteo, rolling stats).

3. **Baseline 3 — Forecasting-based**: LightGBM regressor che predice il
   conteggio atteso; score = scarto standardizzato dal valore predetto.

4. **Sistema ensemble**: combinazione per massimo dei punteggi normalizzati
   (rank-based normalization), con threshold operativo calibrato sul training.

Ogni baseline ritorna un DataFrame con almeno le colonne:
    [chiave, timestamp, score, alert]

dove `score` è in [0, 1] (normalizzato globalmente per confrontabilità).

Riferimenti
-----------
- Cleveland et al. (1990) per STL
- Liu, Ting & Zhou (2008) per Isolation Forest
- Hyndman & Athanasopoulos cap. 5 per forecasting-based detection
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import STL

from src.config import (
    ISOLATION_FOREST_CONTAMINATION,
    N_ESTIMATORS,
    PERCENTILE_SOGLIA_ENSEMBLE,
    SEED,
    SOGLIA_ACCURATEZZA_ALERT,
    STL_RESIDUAL_THRESHOLD,
)


# =============================================================================
# Utility comuni
# =============================================================================
def _normalizza_rank(s: pd.Series) -> pd.Series:
    """Normalizzazione rank-based: converte una serie qualsiasi in [0, 1].

    PERCHÉ rank-based e non min-max o z-score?
    -------------------------------------------
    I tre score baseline hanno distribuzioni molto diverse (residuo standardizzato
    può essere unbounded, IsolationForest dà score in [0, 1] ma con scala
    arbitraria). Una min-max normalization sarebbe instabile (basta un outlier
    enorme a "schiacciare" tutto il resto). La normalizzazione per rango è
    invariante per trasformazioni monotone — confronta gli score sulla base
    della loro posizione nell'ordinamento, non del loro valore assoluto.

    Risultato: ogni baseline produce uno score in [0, 1] dove 1 = osservazione
    più anomala secondo quella baseline. Confrontabili tra loro.
    """
    return s.rank(method="average", pct=True)


def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation: stimatore robusto di scala.

    Definizione: MAD(x) = median(|x_i - median(x)|)

    Per dati gaussiani vale (asintoticamente): σ ≈ 1.4826 × MAD.
    A differenza della std, il MAD non è influenzato dagli outlier
    (breakdown point del 50%), il che lo rende ideale come scala di
    riferimento PROPRIO quando i nostri dati contengono outlier
    (cioè le anomalie che vogliamo trovare).

    Teoria
    ------
    Z-score classico:  z_i = (x_i - mean(x)) / std(x)
    Z-score robusto:   z_i = (x_i - median(x)) / (1.4826 × MAD(x))
    """
    x = np.asarray(x)
    return float(np.median(np.abs(x - np.median(x))))


# =============================================================================
# BASELINE 1 — STL + MAD su residui
# =============================================================================
@dataclass
class ParametriSTL:
    """Iperparametri dell'STL.

    period
        Periodicità della stagionalità *primaria*. Per il traffico orario di
        Bologna usiamo 168 (= 24h × 7 giorni), cioè il ciclo settimanale.
        STL gestisce una sola stagionalità per chiamata; quella giornaliera
        emerge automaticamente come componente del ciclo settimanale.
    robust
        Se True usa loess robust (downweight degli outlier). CRUCIALE per il
        nostro use case: senza robust, le anomalie influenzano l'estrazione
        della stagionalità rendendo il residuo meno informativo.
    """
    period: int = 168
    robust: bool = True


def baseline_stl_mad(
    df: pd.DataFrame,
    parametri: ParametriSTL | None = None,
    soglia_residuo: float = STL_RESIDUAL_THRESHOLD,
    accuratezza_min: float = SOGLIA_ACCURATEZZA_ALERT,
) -> pd.DataFrame:
    """Baseline 1 — Decomposizione STL + soglia MAD.

    Per ciascuna spira, applica STL alla serie storica oraria e calcola
    un residuo standardizzato robusto. Le ore con residuo "estremo"
    diventano alert.

    Algoritmo (per ogni spira indipendentemente)
    ---------------------------------------------
    1. Estrae la serie storica `y_t` ordinata cronologicamente.
    2. STL(y_t, period=168, robust=True) → componenti `trend`, `season`, `resid`.
    3. Calcola median e MAD dei residui.
    4. `z_i = (resid_i - median(resid)) / (1.4826 × MAD(resid))`
    5. `alert_i = |z_i| > soglia_residuo`

    Cosa cattura naturalmente
    -------------------------
    - Anomalie PUNTUALI estreme (singoli outlier): ✓
    - Anomalie contestuali date dalla stagionalità SETTIMANALE: ✓
    - Anomalie collettive brevi (1-3 ore): parzialmente
    - Drift trend lenti: NO (assorbiti dal componente trend)
    - Anomalie dipendenti dal meteo: NO (STL non vede il meteo)

    Ipotesi sui dati
    ----------------
    1. Stagionalità additiva (non moltiplicativa): vero in media, può essere
       imperfetto su spire con grande variabilità di scala.
    2. Residui i.i.d. (stessa distribuzione su tutte le ore): FALSO — il
       traffico è eteroschedastico (varianza dipende dall'ora). Mitigazione
       parziale via MAD ma non risolutiva. → motivo per cui serve l'ensemble.
    """
    parametri = parametri or ParametriSTL()
    risultati: list[pd.DataFrame] = []

    for chiave, gruppo in df.groupby("chiave", sort=False):
        # Ordina e indicizza per tempo (richiesto da STL).
        # Usiamo solo dati affidabili (accuratezza ≥ soglia) per la decomposizione.
        gruppo_ok = (
            gruppo[gruppo["accuratezza"] >= accuratezza_min]
            .set_index("timestamp")["conteggio_veicoli"]
            .astype(float)
            .sort_index()
        )

        # STL richiede una serie senza NaN e con almeno 2 periodi di osservazioni.
        gruppo_ok = gruppo_ok.dropna()
        if len(gruppo_ok) < 2 * parametri.period:
            # Serie troppo corta per STL: emettiamo score = 0 (nessun alert).
            score_default = pd.DataFrame({
                "chiave": chiave,
                "timestamp": gruppo["timestamp"].values,
                "score_b1": 0.0,
                "alert_b1": False,
            })
            risultati.append(score_default)
            continue

        # Reindex su griglia oraria continua per gestire eventuali gap.
        # STL preferisce serie senza buchi; usiamo ffill come imputation
        # minima — i punti imputati non genereranno alert perché il loro
        # residuo sarà ~0 dopo decomposizione.
        idx = pd.date_range(gruppo_ok.index.min(), gruppo_ok.index.max(), freq="h")
        serie = gruppo_ok.reindex(idx).ffill()

        try:
            stl_fit = STL(serie, period=parametri.period, robust=parametri.robust).fit()
        except Exception as exc:
            print(f"  STL fallito su chiave {chiave}: {exc} — fallback score=0")
            score_default = pd.DataFrame({
                "chiave": chiave,
                "timestamp": gruppo["timestamp"].values,
                "score_b1": 0.0,
                "alert_b1": False,
            })
            risultati.append(score_default)
            continue

        # Residuo standardizzato robusto: z_robusto = (r - median) / (1.4826 * MAD)
        residuo = stl_fit.resid.dropna()
        med = float(np.median(residuo))
        mad = _mad(residuo)
        scala = 1.4826 * mad if mad > 0 else 1.0  # protezione divisione per 0
        z_robusto = (residuo - med) / scala
        score = z_robusto.abs()  # usiamo |z| — anomalie in entrambe le direzioni

        # Costruzione dataframe risultato per questa spira.
        out = pd.DataFrame({
            "chiave": chiave,
            "timestamp": score.index,
            "score_b1": score.values,
            "alert_b1": (score.values > soglia_residuo),
        })
        risultati.append(out)

    df_score = pd.concat(risultati, ignore_index=True)
    # Merge col dataset originale per allineamento (timestamp che non hanno
    # avuto STL ricevono score NaN che convertiamo a 0).
    df_score = (
        df[["chiave", "timestamp", "accuratezza"]]
        .merge(df_score, on=["chiave", "timestamp"], how="left")
    )
    df_score["score_b1"] = df_score["score_b1"].fillna(0.0)
    df_score["alert_b1"] = df_score["alert_b1"].fillna(False) & (
        df_score["accuratezza"] >= accuratezza_min
    )
    return df_score[["chiave", "timestamp", "score_b1", "alert_b1"]]


# =============================================================================
# BASELINE 2 — IsolationForest su feature engineered
# =============================================================================
FEATURE_CONTESTUALI = [
    "ora",
    "dow",
    "mese",
    "weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]


def baseline_isolation_forest(
    df: pd.DataFrame,
    feature: list[str] = None,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    seed: int = SEED,
    accuratezza_min: float = SOGLIA_ACCURATEZZA_ALERT,
    train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Baseline 2 — Isolation Forest.

    Idea (Liu, Ting & Zhou 2008): le anomalie sono **facili da isolare**.
    Un albero binario casuale che partiziona ricorsivamente lo spazio
    feature isola un'osservazione anomala in pochi split, mentre un punto
    "normale" richiede molti split per essere isolato. La profondità media
    in foresta di alberi → score di anomalia.

    Score = `-decision_function`: più alto = più anomalo. Lo invertiamo
    in modo che il "verso" sia coerente con le altre baseline.

    Cosa cattura naturalmente
    -------------------------
    - Anomalie CONTESTUALI: ✓ — il modello vede ora, giorno, meteo, lag
    - Combinazioni anomale di feature: ✓ (es. ora 8, pioggia=0, traffico=0
      è una combinazione "strana" anche se ciascuna feature singola sarebbe ok)
    - Anomalie puntuali estreme: ✓ (come caso degenere)
    - Anomalie collettive: parzialmente (ogni ora viene scored singolarmente,
      la coerenza temporale si vede solo a posteriori)

    Ipotesi
    -------
    1. Le anomalie sono **rare** (giustifica `contamination=0.02`)
    2. Le anomalie hanno **valori atipici** nello spazio feature (non solo
       atipici "in senso temporale")
    3. La distribuzione delle feature è stazionaria nel tempo (drift =
       degradazione delle performance)
    """
    feature = feature or FEATURE_CONTESTUALI
    df = df.copy()

    # Selezione del training set: di default tutto il dataset, ma è possibile
    # passare un boolean mask per fittare solo sul periodo storico
    # (es. training temporale).
    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)

    # Filter sul training: solo dati affidabili E nessuna feature NaN.
    df_train = df.loc[train_mask & (df["accuratezza"] >= accuratezza_min)].dropna(
        subset=feature
    )

    if len(df_train) < 100:
        raise ValueError(
            f"Training set troppo piccolo ({len(df_train)} righe) per IsolationForest"
        )

    print(f"  IF training: {len(df_train)} righe, {len(feature)} feature")

    iforest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(df_train[feature].values)

    # Scoring sull'intero dataset (anche test).
    # Le righe con feature NaN ricevono score=0 di default.
    score = pd.Series(0.0, index=df.index)
    valid = df[feature].notna().all(axis=1)
    score.loc[valid] = -iforest.decision_function(df.loc[valid, feature].values)

    df["score_b2"] = score
    # Alert: contamination operativa = soglia automatica scelta dal modello.
    df["alert_b2"] = (
        (df["score_b2"] > np.quantile(score[valid], 1 - contamination))
        & (df["accuratezza"] >= accuratezza_min)
    )

    return df[["chiave", "timestamp", "score_b2", "alert_b2"]]


# =============================================================================
# BASELINE 3 — Forecasting-based (LightGBM)
# =============================================================================
def baseline_forecasting_lgbm(
    df: pd.DataFrame,
    feature: list[str] = None,
    train_mask: pd.Series | None = None,
    seed: int = SEED,
    accuratezza_min: float = SOGLIA_ACCURATEZZA_ALERT,
) -> pd.DataFrame:
    """Baseline 3 — Forecasting + scarto standardizzato.

    Pipeline
    --------
    1. Addestra un LightGBM regressor su (feature contestuali → conteggio).
    2. Predice il conteggio atteso per ciascuna ora-spira.
    3. Calcola `residuo = osservato - predetto`.
    4. Stima `sigma_residuo` (std dei residui sul training set per spira).
    5. Score = `|residuo| / sigma_residuo` (residuo standardizzato).
    6. Alert se score > soglia (default: residui > 3.5 std).

    Cosa cattura
    ------------
    - Anomalie CONTESTUALI nel senso più puro: il modello ha "appreso"
      cosa attendersi date le condizioni → lo scarto è anomalia per costruzione
    - Beneficia di TUTTI i segnali (lag, meteo, calendario, rolling) insieme
    - Tipicamente l'approccio più accurato sui pattern complessi

    Limiti
    ------
    - Richiede un training set "pulito": se ci sono anomalie nel training,
      il modello impara a prevederle (le considera "normali")
    - Su spire con dati molto rumorosi sigma è grande → soglia poco sensibile
    - Sensibile al drift: se il pattern di traffico cambia, le predizioni
      diventano sistematicamente sbagliate (residui sistematicamente alti)
    """
    feature = feature or FEATURE_CONTESTUALI
    df = df.sort_values(["chiave", "timestamp"]).copy()

    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)

    # Training: solo dati affidabili, no NaN nelle feature.
    df_train = df.loc[train_mask & (df["accuratezza"] >= accuratezza_min)].dropna(
        subset=feature + ["conteggio_veicoli"]
    )

    if len(df_train) < 100:
        raise ValueError(f"Training set troppo piccolo: {len(df_train)}")

    print(f"  LGBM training: {len(df_train)} righe, {len(feature)} feature")

    # LGBM regressor con early stopping disabilitato (no validation set qui;
    # la validazione la fa il protocollo di Fase 3).
    # Iperparametri scelti conservativi: profondità limitata, sample rate < 1
    # per ridurre overfit (le anomalie sono rare ma il modello può memorizzarle).
    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(df_train[feature], df_train["conteggio_veicoli"])

    # Predizione su tutto il dataset.
    valid = df[feature].notna().all(axis=1)
    pred = np.full(len(df), np.nan)
    pred[valid.values] = model.predict(df.loc[valid, feature].values)
    df["predizione"] = pred
    df["residuo"] = df["conteggio_veicoli"] - df["predizione"]

    # Stima sigma per spira sul training (più robusta della stima globale).
    sigma_per_spira = (
        df.loc[train_mask & valid]
        .groupby("chiave")["residuo"]
        .std()
        .rename("sigma")
    )
    df = df.merge(sigma_per_spira, on="chiave", how="left")
    # Protezione divisione per 0 (spira con varianza nulla = anomalia di dato).
    df["sigma"] = df["sigma"].replace(0, np.nan).fillna(df["sigma"].median())

    df["score_b3"] = (df["residuo"].abs() / df["sigma"]).fillna(0.0)
    df["alert_b3"] = (df["score_b3"] > 3.5) & (df["accuratezza"] >= accuratezza_min)

    # Salviamo anche la MAE sul training come metrica di qualità del modello.
    df_train_valid = df_train.dropna(subset=feature)
    train_pred = model.predict(df_train_valid[feature])
    mae = mean_absolute_error(df_train_valid["conteggio_veicoli"], train_pred)
    print(f"  LGBM training MAE: {mae:.1f} veicoli/h")

    return df[["chiave", "timestamp", "score_b3", "alert_b3", "predizione", "residuo"]]


# =============================================================================
# SISTEMA ENSEMBLE — max dei punteggi normalizzati
# =============================================================================
def ensemble_max(
    df_b1: pd.DataFrame,
    df_b2: pd.DataFrame,
    df_b3: pd.DataFrame,
    percentile_soglia: float = PERCENTILE_SOGLIA_ENSEMBLE,
) -> pd.DataFrame:
    """Combina i tre punteggi in uno score finale.

    Strategia: rank-normalization indipendente di ogni score + max.

    Razionale (vedi PIANO.md sez. 3.5)
    -----------------------------------
    - **Max** (non media): basta che UNO degli approcci consideri l'osservazione
      anomala per generare un alert. Ottimizza per recall complessivo.
    - **Rank-normalization**: i tre score hanno scale e distribuzioni diverse.
      Convertirli in percentili li rende confrontabili.
    - **Soglia su percentile training**: produce un volume di alert prevedibile
      (X% delle ore-spira) e calibrato sul dominio.

    Alert finale = score_ensemble > soglia operativa.
    """
    # Outer join per allinearli tutti su (chiave, timestamp).
    df = df_b1.merge(df_b2, on=["chiave", "timestamp"], how="outer")
    df = df.merge(df_b3, on=["chiave", "timestamp"], how="outer")

    # Normalizzazione rank per ciascuno score.
    df["score_b1_norm"] = _normalizza_rank(df["score_b1"].fillna(0))
    df["score_b2_norm"] = _normalizza_rank(df["score_b2"].fillna(0))
    df["score_b3_norm"] = _normalizza_rank(df["score_b3"].fillna(0))

    # Ensemble: max dei tre score normalizzati.
    df["score_ensemble"] = df[
        ["score_b1_norm", "score_b2_norm", "score_b3_norm"]
    ].max(axis=1)

    # Calibrazione soglia sul percentile dei punteggi non-nulli.
    soglia = np.percentile(df["score_ensemble"].dropna(), percentile_soglia)
    print(f"  Soglia ensemble (percentile {percentile_soglia}): {soglia:.4f}")

    df["alert_ensemble"] = df["score_ensemble"] > soglia
    df.attrs["soglia_ensemble"] = soglia
    return df


# =============================================================================
# Smoke test (eseguibile come script)
# =============================================================================
if __name__ == "__main__":
    from src.preprocessing import carica_master

    df = carica_master()
    # Smoke test su una porzione (prime 2 spire, primo mese).
    spire = df["chiave"].unique()[:2]
    df_test = df[
        df["chiave"].isin(spire) & (df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=60))
    ].copy()
    print(f"Smoke test su {len(df_test)} righe, {df_test['chiave'].nunique()} spire")

    print("\n[B1] STL+MAD")
    b1 = baseline_stl_mad(df_test)
    print(b1.describe())

    print("\n[B2] IsolationForest")
    b2 = baseline_isolation_forest(df_test)
    print(b2.describe())

    print("\n[B3] Forecasting LGBM")
    b3 = baseline_forecasting_lgbm(df_test)
    print(b3.describe())

    print("\n[Ensemble]")
    ens = ensemble_max(b1, b2, b3)
    print(f"  alert ratio: {ens['alert_ensemble'].mean():.4f}")
