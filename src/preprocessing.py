"""Preprocessing dei dati grezzi → master dataset analitico.

Pipeline di trasformazione che parte dai tre dataset grezzi
(flussi, accuratezza, meteo) e produce un unico DataFrame in formato long
con una riga per ogni `(spira, timestamp)` arricchita con feature contestuali.

Pipeline a 6 stadi
------------------
1. **Melt flussi** : da formato wide (24 colonne orarie) a long
2. **Melt accuratezza** : stesso pattern, schema leggermente diverso
3. **Selezione spire** : 15-20 sensori stratificati per funzione urbana
4. **Merge** : flussi ⊕ accuratezza ⊕ meteo
5. **Calendario** : festività nazionali + locali bolognesi → `tipo_giorno`
6. **Feature engineering** : lag, rolling stats, encoding categoriche

Riferimenti alla traccia
------------------------
- Sez. 3.1 "Operazione preliminare": il melt è il primo step esplicitamente richiesto
- Sez. 3.2: gestione semantica dell'accuratezza nel join
- Sez. 4 "Vincoli pratici": selezione di 15-25 spire stratificate
"""

from __future__ import annotations

import re

import holidays
import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    FESTIVITA_LOCALI_BOLOGNA,
    N_SPIRE_TARGET,
    SOGLIA_ACCURATEZZA_AFFIDABILE,
    SOGLIA_ACCURATEZZA_MINIMA,
)


# =============================================================================
# 1. MELT FLUSSI: wide → long
# =============================================================================
# Regex per identificare le 24 colonne orarie del dataset flussi.
# Pattern: HH_00_HH_00 (4 segmenti di 2 cifre). Es: '08_00_09_00'.
_PAT_ORE_FLUSSI = re.compile(r"^\d{2}_\d{2}_\d{2}_\d{2}$")


def melt_flussi(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Trasforma il dataset flussi da formato wide a long.

    Input wide: una riga per (spira, giorno), 24 colonne orarie.
    Output long: una riga per (spira, ora), con colonna `timestamp` e
    `conteggio_veicoli`.

    Colonne id_vars preservate (replicate per ogni ora):
      - data, id_uni, chiave : identificatori
      - nome_via, direzione, longitudine, latitudine : metadati (se presenti)
    """
    colonne_orarie = [c for c in df_wide.columns if _PAT_ORE_FLUSSI.match(c)]
    if len(colonne_orarie) != 24:
        raise ValueError(
            f"Atteso 24 colonne orarie, trovate {len(colonne_orarie)} "
            "— verificare schema dataset"
        )

    # id_vars di base + metadati opzionali (presenti se non filtrati a monte).
    id_vars = ["data", "id_uni", "chiave"]
    for opt in ["nome_via", "direzione", "longitudine", "latitudine"]:
        if opt in df_wide.columns:
            id_vars.append(opt)

    df_long = df_wide.melt(
        id_vars=id_vars,
        value_vars=colonne_orarie,
        var_name="fascia_oraria",
        value_name="conteggio_veicoli",
    )

    # Costruzione timestamp orario.
    # I primi 2 caratteri della fascia ('08_00_09_00' → '08') sono l'ora di inizio.
    df_long["ora"] = df_long["fascia_oraria"].str[:2].astype(int)
    df_long["timestamp"] = (
        pd.to_datetime(df_long["data"]) + pd.to_timedelta(df_long["ora"], unit="h")
    )

    # Coercizione robusta a numerico (NaN su valori malformati).
    df_long["conteggio_veicoli"] = pd.to_numeric(
        df_long["conteggio_veicoli"], errors="coerce"
    )

    # Downcast a int32: conteggi orari raramente superano i 5000, int32 basta
    # e dimezza il consumo di memoria rispetto a int64.
    # Cast a float32 (anziché Int32 nullable) per evitare friction con
    # matplotlib/seaborn (che richiedono dtype numpy nativi per il plotting).
    df_long["conteggio_veicoli"] = df_long["conteggio_veicoli"].astype("float32")

    return df_long.drop(columns=["fascia_oraria", "ora"])


# =============================================================================
# 2. MELT ACCURATEZZA
# =============================================================================
# Pattern diverso: HH_00_HH (3 segmenti). Es: '08_00_09'.
_PAT_ORE_ACC = re.compile(r"^\d{2}_\d{2}_\d{2}$")


def melt_accuratezza(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Trasforma il dataset accuratezza in formato long.

    Gestisce due peculiarità del dataset:
      - valori in formato stringa '85%' → conversione a float
      - sentinel -1 (e altri valori negativi) → NaN
    """
    colonne_orarie = [c for c in df_wide.columns if _PAT_ORE_ACC.match(c)]
    if len(colonne_orarie) != 24:
        raise ValueError(
            f"Atteso 24 colonne orarie accuratezza, trovate {len(colonne_orarie)}"
        )

    df_long = df_wide.melt(
        id_vars=["data_2", "chiave"],
        value_vars=colonne_orarie,
        var_name="fascia_oraria_acc",
        value_name="accuratezza_str",
    )

    df_long["ora"] = df_long["fascia_oraria_acc"].str[:2].astype(int)
    df_long["timestamp"] = (
        pd.to_datetime(df_long["data_2"]) + pd.to_timedelta(df_long["ora"], unit="h")
    )

    # Parsing valore: '85%' → 85.0. `replace('', np.nan)` cattura eventuali
    # stringhe vuote che, dopo lo strip del '%', diventerebbero crash su astype.
    df_long["accuratezza"] = (
        df_long["accuratezza_str"]
        .astype(str)
        .str.rstrip("%")
        .replace("", np.nan)
        .astype(float)
    )

    # Sentinel: valori negativi = "dato non disponibile" → NaN.
    # Semantica diversa da 0 (sensore presente ma totalmente offline).
    df_long.loc[df_long["accuratezza"] < 0, "accuratezza"] = np.nan

    return df_long[["chiave", "timestamp", "accuratezza"]]


# =============================================================================
# 3. SELEZIONE SPIRE STRATIFICATA
# =============================================================================
def seleziona_spire_top_qualita(
    df_flussi_long: pd.DataFrame,
    df_acc_long: pd.DataFrame,
    n_spire: int = N_SPIRE_TARGET,
) -> list[int]:
    """Seleziona le N spire con la maggiore qualità del dato nel periodo.

    Criterio "qualità": numero di ore con accuratezza ≥ soglia affidabile.
    Questa è una selezione "ragionata sul dato" che produce un sottoinsieme
    di sensori utilizzabili. NON è una stratificazione geografica completa
    (che richiederebbe domain knowledge urbano) — viene affinata in EDA.

    Ritorna la lista di `chiave` (ID sensore fisico) selezionate.
    """
    # Conteggio ore di alta qualità per ciascun sensore.
    qualita = (
        df_acc_long.assign(
            affidabile=df_acc_long["accuratezza"] >= SOGLIA_ACCURATEZZA_AFFIDABILE
        )
        .groupby("chiave")["affidabile"]
        .sum()
        .sort_values(ascending=False)
    )

    # Manteniamo solo chiavi effettivamente presenti anche nei flussi.
    chiavi_in_flussi = set(df_flussi_long["chiave"].dropna().unique())
    qualita = qualita[qualita.index.isin(chiavi_in_flussi)]

    selezionate = qualita.head(n_spire).index.tolist()
    print(
        f"Selezionate {len(selezionate)} spire su {len(qualita)} candidate. "
        f"Range ore affidabili: {qualita.iloc[:len(selezionate)].min():.0f}—"
        f"{qualita.iloc[:len(selezionate)].max():.0f}"
    )
    return selezionate


# =============================================================================
# 4. CALENDARIO E TIPO GIORNO
# =============================================================================
def aggiungi_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge feature di calendario al DataFrame in formato long.

    Feature derivate:
      - `ora`               : 0..23
      - `giorno_settimana`  : nome inglese (Monday, ..., Sunday)
      - `dow`               : 0..6 (lunedì=0)
      - `mese`              : 1..12
      - `weekend`           : bool (sab/dom)
      - `festivo_nazionale` : bool (festività italiane)
      - `nome_festivita`    : str | None
      - `festa_locale`      : bool (calendario Bologna)
      - `tipo_giorno`       : 'feriale' | 'weekend' | 'festivo'

    Il `tipo_giorno` è la feature più rilevante per il framing: separa i
    tre regimi di traffico che richiedono modelli distinti.
    """
    df = df.copy()

    df["ora"] = df["timestamp"].dt.hour
    df["giorno_settimana"] = df["timestamp"].dt.day_name()
    df["dow"] = df["timestamp"].dt.dayofweek
    df["mese"] = df["timestamp"].dt.month
    df["weekend"] = df["dow"] >= 5

    # Festività nazionali italiane via pkg `holidays`.
    anni = sorted(df["timestamp"].dt.year.unique().tolist())
    festivita_it = holidays.country_holidays("IT", years=anni)

    data_giorno = df["timestamp"].dt.date
    df["festivo_nazionale"] = data_giorno.map(lambda d: d in festivita_it).astype(bool)
    df["nome_festivita"] = data_giorno.map(lambda d: festivita_it.get(d))

    # Festività locali bolognesi: confronto su MM-DD (per applicare la stessa
    # festività a tutti gli anni del dataset).
    mmdd = df["timestamp"].dt.strftime("%m-%d")
    df["festa_locale"] = mmdd.isin(FESTIVITA_LOCALI_BOLOGNA.keys())

    # Categorizzazione finale.
    # Ordine dei branch (festivo > weekend > feriale): un Natale che cade di
    # sabato è semanticamente più "festivo" che "weekend" per il modeling.
    condizioni = [
        df["festivo_nazionale"] | df["festa_locale"],
        df["weekend"],
    ]
    scelte = ["festivo", "weekend"]
    df["tipo_giorno"] = np.select(condizioni, scelte, default="feriale")

    return df


# =============================================================================
# 5. FEATURE ENGINEERING per il modeling
# =============================================================================
def aggiungi_lag_features(
    df: pd.DataFrame,
    col_target: str = "conteggio_veicoli",
    lags: tuple[int, ...] = (1, 24, 168),
) -> pd.DataFrame:
    """Aggiunge feature di lag per il forecasting.

    Lag rilevanti per il traffico:
      - 1h    : autocorrelazione immediata
      - 24h   : stessa ora del giorno prima (ciclo giornaliero)
      - 168h  : stessa ora della settimana prima (ciclo settimanale)

    IMPORTANTE: i lag sono calcolati PER SPIRA. Usiamo `groupby('chiave')`
    + `shift()` per evitare "leakage" tra sensori diversi.
    """
    df = df.sort_values(["chiave", "timestamp"]).copy()
    for lag in lags:
        col_name = f"lag_{lag}h"
        df[col_name] = df.groupby("chiave")[col_target].shift(lag)
    return df


def aggiungi_rolling_features(
    df: pd.DataFrame,
    col_target: str = "conteggio_veicoli",
    windows: tuple[int, ...] = (24,),
) -> pd.DataFrame:
    """Statistiche rolling (media, std) sulle ultime N ore per spira.

    Catturano lo "stato locale" del traffico: una spira che nelle ultime
    24 ore ha avuto media molto bassa è in un regime diverso da una
    spira con media alta — questa info aiuta il modello a calibrare il
    valore "atteso" indipendentemente dalle assolute della spira.

    Anti-leakage: lo shift(1) garantisce che la finestra termini all'ora
    PRIMA del timestamp corrente — il modello non vede mai il valore stesso.
    """
    df = df.sort_values(["chiave", "timestamp"]).copy()
    for w in windows:
        shifted = df.groupby("chiave")[col_target].shift(1)
        df[f"rolling_mean_{w}h"] = (
            shifted.groupby(df["chiave"]).rolling(w, min_periods=1)
            .mean().reset_index(level=0, drop=True)
        )
        df[f"rolling_std_{w}h"] = (
            shifted.groupby(df["chiave"]).rolling(w, min_periods=1)
            .std().reset_index(level=0, drop=True)
        )
    return df


# =============================================================================
# 6. PIPELINE END-TO-END
# =============================================================================
def costruisci_master_dataset(
    df_flussi_raw: pd.DataFrame,
    df_acc_raw: pd.DataFrame,
    df_meteo: pd.DataFrame,
    n_spire: int = N_SPIRE_TARGET,
    salva: bool = True,
) -> pd.DataFrame:
    """Pipeline end-to-end: dataset grezzi → master analitico.

    Steps eseguiti in ordine:
      1. Melt flussi & accuratezza
      2. Selezione N spire (top qualità)
      3. Filtro dataset alle spire selezionate
      4. Merge flussi ⊕ accuratezza ⊕ meteo
      5. Calendario (festività, tipo_giorno)
      6. Feature lag + rolling
      7. Flag `dato_affidabile` (accuratezza ≥ soglia)

    Output: DataFrame indicizzato su (chiave, timestamp), pronto per
    l'EDA e il modeling.
    """
    print("[preprocessing] 1/6 melt flussi")
    df_flussi = melt_flussi(df_flussi_raw)

    print("[preprocessing] 2/6 melt accuratezza")
    df_acc = melt_accuratezza(df_acc_raw)

    print("[preprocessing] 3/6 selezione spire")
    spire = seleziona_spire_top_qualita(df_flussi, df_acc, n_spire=n_spire)
    df_flussi = df_flussi[df_flussi["chiave"].isin(spire)].copy()
    df_acc = df_acc[df_acc["chiave"].isin(spire)].copy()

    print("[preprocessing] 4/6 merge flussi+accuratezza+meteo")
    df = df_flussi.merge(df_acc, on=["chiave", "timestamp"], how="left")
    df = df.merge(df_meteo, on="timestamp", how="left")

    print("[preprocessing] 5/6 calendario")
    df = aggiungi_calendario(df)

    print("[preprocessing] 6/6 feature lag + rolling")
    df = aggiungi_lag_features(df)
    df = aggiungi_rolling_features(df)

    # Flag operativo di qualità dato (vedi PIANO.md sez. 3.3).
    df["dato_affidabile"] = df["accuratezza"] >= SOGLIA_ACCURATEZZA_AFFIDABILE
    df["dato_utilizzabile"] = df["accuratezza"] >= SOGLIA_ACCURATEZZA_MINIMA

    # Ordine canonico per ispezione e modeling.
    df = df.sort_values(["chiave", "timestamp"]).reset_index(drop=True)

    if salva:
        out_path = DATA_PROCESSED_DIR / "master.parquet"
        df.to_parquet(out_path)
        print(f"[preprocessing] salvato master dataset in {out_path}")
        print(f"  shape={df.shape}, spire={df['chiave'].nunique()}, "
              f"periodo={df['timestamp'].min()} → {df['timestamp'].max()}")

    return df


def carica_master(force_rebuild: bool = False) -> pd.DataFrame:
    """Carica il master dataset, ricostruendolo se necessario.

    Idempotente: se `data/processed/master.parquet` esiste, lo legge.
    Altrimenti, scarica i dati grezzi e ricostruisce.
    """
    from src.config import ANNI_DA_SCARICARE, DATA_INIZIO, DATA_FINE
    from src.data_loading import carica_accuratezza, carica_flussi, carica_meteo

    master_path = DATA_PROCESSED_DIR / "master.parquet"
    if master_path.exists() and not force_rebuild:
        print(f"[master] da cache ({master_path.name})")
        return pd.read_parquet(master_path)

    print("[master] rebuild in corso...")
    df_flussi = carica_flussi(ANNI_DA_SCARICARE)
    df_acc = carica_accuratezza(ANNI_DA_SCARICARE)
    df_meteo = carica_meteo(DATA_INIZIO, DATA_FINE)
    return costruisci_master_dataset(df_flussi, df_acc, df_meteo)


if __name__ == "__main__":
    # Smoke test: ricostruisce il master dataset.
    df = carica_master(force_rebuild=False)
    print("\n=== Sample ===")
    print(df.head())
    print("\n=== Dtypes ===")
    print(df.dtypes)
