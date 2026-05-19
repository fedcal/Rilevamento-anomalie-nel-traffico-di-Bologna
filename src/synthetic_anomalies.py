"""Iniezione di anomalie sintetiche per la validazione del sistema.

In assenza di etichette di ground truth, la traccia (sez. 3 Fase 3.b)
suggerisce diverse strategie di valutazione. Questo modulo implementa la
**prima strategia**: iniezione di anomalie sintetiche di cui conosciamo la
posizione e la natura, su cui poi misuriamo la capacità del sistema di
recuperarle.

Tassonomia delle anomalie (PIANO.md sez. 4 Fase 3b)
----------------------------------------------------
1. **Spike**     : moltiplicatore × valore atteso (+200% o +400%)
2. **Drop**      : conteggio drasticamente ridotto (-90% del valore atteso)
3. **Zero Plateau** : conteggio = 0 per N ore consecutive (con accuratezza buona)
4. **Shift**     : valore costante (+50%) per N ore consecutive

CAVEAT importante (traccia sez. 8)
-----------------------------------
"Costruirsi un test set troppo facile iniettando anomalie sintetiche enormi
e ovvie, ottenere recall del 100%, e dichiararsi soddisfatti" è uno dei
fallimenti più comuni. Per evitarlo:
  - iniettiamo anomalie BORDERLINE (es. +50% non +500%)
  - includiamo varianti di durata realistica (1 ora, 3 ore, 6 ore)
  - randomizziamo i punti di iniezione su tipologie di spira diverse
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.config import SEED

TipoAnomalia = Literal["spike", "drop", "zero_plateau", "shift"]


@dataclass
class AnomaliaSintetica:
    """Record che descrive una singola anomalia iniettata."""
    chiave: int
    timestamp_inizio: pd.Timestamp
    durata_ore: int
    tipo: TipoAnomalia
    intensita: float  # moltiplicatore (per spike/drop/shift) o irrilevante per zero
    valore_originale: list[float]
    valore_iniettato: list[float]


def _calcola_valore_iniettato(
    valore_origin: np.ndarray,
    tipo: TipoAnomalia,
    intensita: float,
) -> np.ndarray:
    """Calcola il valore "iniettato" in funzione del tipo e dell'intensità.

    - spike(I):       v' = v * (1 + I)
    - drop(I):        v' = v * I              (I < 1; es. 0.1 = -90%)
    - shift(I):       v' = v * (1 + I)        (come spike ma su una sequenza)
    - zero_plateau:   v' = 0
    """
    valore_origin = np.asarray(valore_origin, dtype=float)
    if tipo == "spike":
        return valore_origin * (1.0 + intensita)
    if tipo == "drop":
        return valore_origin * intensita
    if tipo == "shift":
        return valore_origin * (1.0 + intensita)
    if tipo == "zero_plateau":
        return np.zeros_like(valore_origin)
    raise ValueError(f"Tipo anomalia non riconosciuto: {tipo}")


def inietta_anomalie(
    df: pd.DataFrame,
    n_anomalie: int = 100,
    seed: int = SEED,
    finestra_iniezione: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    distribuzione_tipi: dict[TipoAnomalia, float] | None = None,
    intensita_minima: dict[TipoAnomalia, float] | None = None,
) -> tuple[pd.DataFrame, list[AnomaliaSintetica]]:
    """Inietta N anomalie sintetiche in un DataFrame.

    L'iniezione MODIFICA `conteggio_veicoli` in posizioni casuali e ritorna:
      - una nuova copia del DataFrame (immutabilità preservata)
      - la lista delle anomalie iniettate (ground truth proxy)

    Parametri
    ---------
    df : DataFrame in formato long, deve avere colonne (chiave, timestamp,
         conteggio_veicoli, accuratezza)
    n_anomalie : numero target di anomalie da iniettare
    seed : random seed per riproducibilità
    finestra_iniezione : (start, end) timestamp — limita le posizioni candidate
                         (utile per iniettare solo nel test set)
    distribuzione_tipi : dict {tipo: peso}. Default: bilanciata 25% ciascuno.
    intensita_minima : intensità borderline per ciascun tipo. Default:
                       valori che producono anomalie realistiche, non triviali
                       da rilevare.

    Vincoli per evitare iniezioni patologiche
    ------------------------------------------
    1. Solo su ore con dato affidabile (acc ≥ 80%) — non riniettiamo su
       ore già "rotte" dal sensore.
    2. Le sequenze (zero_plateau, shift) non possono attraversare gap del dato.
    3. Niente sovrapposizione tra anomalie iniettate.
    """
    rng = np.random.default_rng(seed)

    distribuzione_tipi = distribuzione_tipi or {
        "spike": 0.30,
        "drop": 0.30,
        "zero_plateau": 0.20,
        "shift": 0.20,
    }
    intensita_minima = intensita_minima or {
        "spike": 1.0,        # +100% (raddoppio)
        "drop": 0.1,         # -90% del valore originale
        "zero_plateau": 0.0, # irrilevante
        "shift": 0.5,        # +50%
    }

    df = df.copy().sort_values(["chiave", "timestamp"]).reset_index(drop=True)

    # Pool di posizioni candidate: solo dati affidabili in finestra.
    mask = df["accuratezza"] >= 80.0
    if finestra_iniezione is not None:
        start, end = finestra_iniezione
        mask &= (df["timestamp"] >= start) & (df["timestamp"] <= end)
    pool_indici = df.index[mask].to_numpy()
    if len(pool_indici) < n_anomalie * 10:
        print(f"WARN: pool di candidati piccolo ({len(pool_indici)}) per {n_anomalie} anomalie")

    # Campiona tipi secondo distribuzione.
    tipi = list(distribuzione_tipi.keys())
    pesi = np.array([distribuzione_tipi[t] for t in tipi])
    pesi = pesi / pesi.sum()
    tipi_estratti = rng.choice(tipi, size=n_anomalie, p=pesi)

    anomalie: list[AnomaliaSintetica] = []
    indici_usati: set[int] = set()

    for tipo in tipi_estratti:
        tipo = str(tipo)
        # Durata: 1h per puntuali (spike/drop), 3-8 ore per collettive.
        if tipo in ("spike", "drop"):
            durata = 1
        elif tipo == "zero_plateau":
            durata = int(rng.integers(3, 9))
        else:  # shift
            durata = int(rng.integers(3, 13))

        # Intensità: borderline (1x-1.5x il minimo) per evitare "anomalie ovvie".
        if tipo == "drop":
            intensita = float(intensita_minima[tipo] * rng.uniform(0.5, 1.5))
            intensita = min(intensita, 0.3)  # cap a -70%
        elif tipo == "zero_plateau":
            intensita = 0.0
        else:
            intensita = float(intensita_minima[tipo] * rng.uniform(1.0, 1.5))

        # Cerca un indice di partenza che permetta `durata` ore consecutive
        # nella stessa spira senza sovrapposizioni.
        candidate = rng.permutation(pool_indici)[:200]  # limite per performance
        scelto = None
        for idx_start in candidate:
            if any((idx_start + d) in indici_usati for d in range(durata)):
                continue
            # Verifica continuità nella stessa spira.
            chiave_start = df.at[idx_start, "chiave"]
            slice_idx = list(range(idx_start, idx_start + durata))
            if max(slice_idx) >= len(df):
                continue
            if not all(df.at[i, "chiave"] == chiave_start for i in slice_idx):
                continue
            # Verifica accuratezza alta su tutto il segmento.
            if not all(df.at[i, "accuratezza"] >= 80.0 for i in slice_idx):
                continue
            scelto = idx_start
            break

        if scelto is None:
            continue  # non siamo riusciti a piazzarla — skip

        slice_idx = list(range(scelto, scelto + durata))
        for i in slice_idx:
            indici_usati.add(i)

        valori_orig = df.loc[slice_idx, "conteggio_veicoli"].astype(float).values
        valori_inj = _calcola_valore_iniettato(valori_orig, tipo, intensita)  # type: ignore[arg-type]
        valori_inj = np.maximum(valori_inj, 0)  # no conteggi negativi

        # Applica l'iniezione (rounding a intero per coerenza col dato originale).
        # Cast a float32 per coerenza col master dataset (conteggio_veicoli è float32).
        df.loc[slice_idx, "conteggio_veicoli"] = np.round(valori_inj).astype("float32")

        anomalie.append(AnomaliaSintetica(
            chiave=int(df.at[scelto, "chiave"]),
            timestamp_inizio=df.at[scelto, "timestamp"],
            durata_ore=durata,
            tipo=tipo,  # type: ignore[arg-type]
            intensita=intensita,
            valore_originale=valori_orig.tolist(),
            valore_iniettato=valori_inj.tolist(),
        ))

    print(f"Iniettate {len(anomalie)}/{n_anomalie} anomalie sintetiche.")
    return df, anomalie


def anomalie_a_dataframe(anomalie: list[AnomaliaSintetica]) -> pd.DataFrame:
    """Converte la lista di AnomaliaSintetica in un DataFrame "ground truth proxy"
    espandendo le sequenze in una riga per ora-spira coinvolta.

    Output: DataFrame con colonne `[chiave, timestamp, tipo, intensita, durata_ore]`,
    una riga per ciascuna ora-spira "vera anomalia" iniettata. Pronto per merge
    con i risultati del modello.
    """
    if not anomalie:
        return pd.DataFrame(columns=["chiave", "timestamp", "tipo", "intensita", "durata_ore"])

    righe = []
    for a in anomalie:
        for offset in range(a.durata_ore):
            righe.append({
                "chiave": a.chiave,
                "timestamp": a.timestamp_inizio + pd.Timedelta(hours=offset),
                "tipo": a.tipo,
                "intensita": a.intensita,
                "durata_ore": a.durata_ore,
            })
    return pd.DataFrame(righe)


if __name__ == "__main__":
    from src.preprocessing import carica_master

    df = carica_master()
    df_iniettato, anomalie = inietta_anomalie(df, n_anomalie=50)
    df_gt = anomalie_a_dataframe(anomalie)
    print(df_gt["tipo"].value_counts())
    print(df_gt.head())
