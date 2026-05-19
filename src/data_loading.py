"""Download e caching dei dataset utilizzati nel project work.

Tre sorgenti dati indipendenti vengono qui esposte con la stessa interfaccia
("scarica oppure leggi da cache"):

1. **Flussi spire Bologna** (Open Data Comune di Bologna)
2. **Accuratezza spire**    (Open Data Comune di Bologna — dataset gemello)
3. **Meteo storico**        (Open-Meteo Historical Weather API)

Pattern di caching
------------------
- I dati grezzi vengono salvati una volta sola in `data/raw/` come Parquet.
- Le funzioni `carica_*` controllano la cache prima di chiamare l'API.
- Il formato Parquet è preferito a CSV: mantiene i dtype, è compresso, e
  l'I/O è ~10x più veloce.

Riferimenti alla traccia
------------------------
- Sez. 3.1: dataset flussi spire (formato wide, 24 colonne orarie)
- Sez. 3.2: dataset accuratezza (necessario per fenomeno vs strumento)
- Sez. 3.3: Open-Meteo (CAVEAT timezone Europa/Roma vs UTC)
- Sez. 4: "Salvate i dati grezzi una volta sola" → pattern read-through cache
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_RAW_DIR, LAT, LON


# =============================================================================
# Utility HTTP — wrapper con retry esponenziale
# =============================================================================
def _get_with_retry(
    url: str,
    params: dict | None = None,
    timeout: int = 180,
    n_retries: int = 3,
    backoff_seconds: float = 5.0,
) -> requests.Response:
    """GET HTTP con retry esponenziale su errori temporanei.

    Le API pubbliche occasionalmente restituiscono 502/503 sotto carico,
    oppure timeout di rete. Un retry ingenuo (`for _ in range(3): get()`)
    sovraccarica ulteriormente il server; usiamo invece un backoff
    esponenziale: 5s, 10s, 20s tra i tentativi.

    Solleva HTTPError sul tentativo finale in caso di fallimento persistente.
    """
    last_err: Exception | None = None
    for attempt in range(n_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            # backoff esponenziale: 5, 10, 20 secondi
            sleep_for = backoff_seconds * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{n_retries}] {type(e).__name__}: dormo {sleep_for}s")
            time.sleep(sleep_for)
    # Se siamo qui, tutti i retry sono falliti
    raise RuntimeError(f"GET fallito dopo {n_retries} tentativi: {last_err}")


# =============================================================================
# 1. FLUSSI SPIRE
# =============================================================================
def _url_flussi(anno: int) -> str:
    """URL dell'endpoint Opendatasoft per i flussi dell'anno richiesto."""
    dataset_id = f"rilevazione-flusso-veicoli-tramite-spire-anno-{anno}"
    return (
        "https://opendata.comune.bologna.it/api/explore/v2.1/"
        f"catalog/datasets/{dataset_id}/exports/json"
    )


def scarica_flussi_anno(anno: int) -> pd.DataFrame:
    """Scarica l'intero dataset annuale dei flussi spire.

    Usiamo l'endpoint `/exports/json` (non `/records`) perché quest'ultimo
    è paginato con limite max offset=10000, valore ampiamente superato dal
    dataset annuale (decine di spire × 365 giorni).

    Ritorna un DataFrame in formato wide: una riga per (spira, giorno),
    con 24 colonne orarie `00_00_01_00`, ..., `23_00_24_00`.
    """
    url = _url_flussi(anno)
    print(f"[flussi {anno}] scaricamento in corso (può richiedere alcuni minuti)...")
    r = _get_with_retry(url, timeout=600)  # 10 min: dataset annuale è grande
    df = pd.DataFrame(r.json())
    print(f"[flussi {anno}] OK — shape={df.shape}")
    return df


def carica_flussi(anni: tuple[int, ...]) -> pd.DataFrame:
    """Carica i flussi di più anni, con cache filesystem per anno.

    Per ogni anno richiesto:
      - Se esiste `data/raw/flussi_<anno>.parquet` → legge dalla cache
      - Altrimenti → scarica, salva in Parquet, restituisce.

    Concatena infine in un unico DataFrame.
    """
    dfs = []
    for anno in anni:
        cache_file = DATA_RAW_DIR / f"flussi_{anno}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            print(f"[flussi {anno}] da cache ({cache_file.name}) — shape={df.shape}")
        else:
            df = scarica_flussi_anno(anno)
            df.to_parquet(cache_file)
            print(f"[flussi {anno}] salvato in {cache_file.name}")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# 2. ACCURATEZZA SPIRE
# =============================================================================
def _url_accuratezza(anno: int) -> str:
    dataset_id = f"accuratezza-spire-anno-{anno}"
    return (
        "https://opendata.comune.bologna.it/api/explore/v2.1/"
        f"catalog/datasets/{dataset_id}/exports/json"
    )


def scarica_accuratezza_anno(anno: int) -> pd.DataFrame:
    """Scarica il dataset annuale di accuratezza per le spire.

    Schema differente dal dataset flussi (vedi cella demo per dettagli):
      - campo data       : 'data_2'
      - campo spira      : 'codice_spira_2'
      - colonne orarie   : 'HH_00_HH' (3 segmenti, non 4)
      - valori           : stringhe '85%' (con simbolo)
      - sentinel         : '-1%' = dato mancante
    """
    url = _url_accuratezza(anno)
    print(f"[accuratezza {anno}] scaricamento in corso...")
    r = _get_with_retry(url, timeout=600)
    df = pd.DataFrame(r.json())
    print(f"[accuratezza {anno}] OK — shape={df.shape}")
    return df


def carica_accuratezza(anni: tuple[int, ...]) -> pd.DataFrame:
    """Carica accuratezza di più anni con cache per anno (vedi `carica_flussi`)."""
    dfs = []
    for anno in anni:
        cache_file = DATA_RAW_DIR / f"accuratezza_{anno}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            print(f"[accuratezza {anno}] da cache ({cache_file.name}) — shape={df.shape}")
        else:
            df = scarica_accuratezza_anno(anno)
            df.to_parquet(cache_file)
            print(f"[accuratezza {anno}] salvato in {cache_file.name}")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# 3. METEO STORICO (Open-Meteo)
# =============================================================================
def scarica_meteo(
    data_inizio: str,
    data_fine: str,
    lat: float = LAT,
    lon: float = LON,
) -> pd.DataFrame:
    """Scarica i dati meteo orari da Open-Meteo per Bologna.

    Variabili scaricate:
      - temperature_2m  : temperatura a 2m (°C)
      - precipitation   : pioggia + neve (mm/h)
      - rain            : solo pioggia (mm/h)
      - snowfall        : solo neve (cm/h)
      - wind_speed_10m  : velocità vento a 10m (km/h)
      - weather_code    : codice WMO categorico

    Timezone — CRUCIALE (vedi traccia, sez. 8 "Trappole comuni")
    --------------------------------------------------------------
    `timezone='Europe/Rome'` esplicito. Open-Meteo di default usa UTC;
    senza override, in inverno avremmo un offset di 1 ora (CET) e
    in estate di 2 ore (CEST). Sui join con i dati spire (in ora locale)
    questo invalida completamente la feature meteo.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": data_inizio,
        "end_date": data_fine,
        "hourly": (
            "temperature_2m,precipitation,rain,snowfall,"
            "wind_speed_10m,weather_code"
        ),
        "timezone": "Europe/Rome",
    }
    r = _get_with_retry(url, params=params, timeout=120)
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["timestamp"] = pd.to_datetime(df["time"])
    return df.drop(columns=["time"])


def carica_meteo(data_inizio: str, data_fine: str) -> pd.DataFrame:
    """Carica il meteo con cache filesystem.

    Nome del file include gli estremi temporali, così cambiando finestra
    si ottiene una cache distinta senza dover invalidare manualmente.
    """
    cache_file = DATA_RAW_DIR / f"meteo_{data_inizio}_{data_fine}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        print(f"[meteo] da cache ({cache_file.name}) — shape={df.shape}")
        return df
    print(f"[meteo] scaricamento {data_inizio} → {data_fine}...")
    df = scarica_meteo(data_inizio, data_fine)
    df.to_parquet(cache_file)
    print(f"[meteo] salvato in {cache_file.name} — shape={df.shape}")
    return df


# =============================================================================
# ENTRYPOINT — eseguibile come script per scaricare tutto in una volta
# =============================================================================
def main() -> None:
    """Scarica tutti i dataset per la finestra di analisi (chiamabile da CLI).

    Uso:
        cd <project_root>
        python -m src.data_loading
    """
    from src.config import ANNI_DA_SCARICARE, DATA_INIZIO, DATA_FINE

    print("=" * 70)
    print(f"Download dati progetto — finestra {DATA_INIZIO} → {DATA_FINE}")
    print("=" * 70)

    print("\n--- 1/3 Flussi spire ---")
    carica_flussi(ANNI_DA_SCARICARE)

    print("\n--- 2/3 Accuratezza spire ---")
    carica_accuratezza(ANNI_DA_SCARICARE)

    print("\n--- 3/3 Meteo Bologna ---")
    carica_meteo(DATA_INIZIO, DATA_FINE)

    print("\nDownload completato.")


if __name__ == "__main__":
    main()
