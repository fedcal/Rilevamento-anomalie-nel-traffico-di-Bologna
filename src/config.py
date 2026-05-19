"""Configurazione globale del project work.

Tutte le costanti che hanno effetto trasversale sui moduli (seed, percorsi,
finestra temporale di analisi, soglie operative) sono raccolte qui.
Modificare un valore in questo file deve propagarsi al resto del progetto
SENZA dover toccare altri moduli — questo file è l'unico "punto di verità"
per i parametri di scenario.

Riferimenti alla traccia
------------------------
- Sez. 4 "Vincoli pratici": seed fissati, periodo gestibile, cache locale
- Sez. 3.2: gestione dataset di accuratezza (soglie qui definite)
- Sez. 5 Fase 1: framing iniziale → scelte operative riflesse qui
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# =============================================================================
# Riproducibilità (CRUCIALE — la traccia lo definisce criterio di valutazione)
# =============================================================================
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fissa i seed delle librerie randomiche utilizzate dal progetto.

    Va chiamata all'inizio di ciascun notebook e di ciascun entry-point
    di script per garantire riproducibilità bit-identica dei risultati.

    Librerie coperte:
      - `random` (stdlib)
      - `numpy.random` (default global RNG)
      - variabile d'ambiente PYTHONHASHSEED (per dizionari/ordinamenti)

    Per librerie con RNG proprio (sklearn, lightgbm) il seed viene
    passato esplicitamente come parametro `random_state` al costruttore.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Percorsi del progetto (relative al root del repository)
# =============================================================================
# `__file__` di questo modulo è `<root>/src/config.py`; risaliamo di 1 livello.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_FIGURES_DIR: Path = RESULTS_DIR / "figures"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Creazione idempotente delle directory all'import.
# Evita errori "no such directory" quando i moduli scrivono file.
for _d in (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RESULTS_DIR,
    RESULTS_FIGURES_DIR,
    DOCS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Finestra temporale di analisi
# =============================================================================
# Vedi PIANO.md sez. 3.1 per la motivazione di questa scelta.
# - Train : 15 mesi → copre Q1+Q2+Q3+Q4 2024 + Q1 2025 (stagionalità completa)
# - Test  : 3 mesi (Q2 2025) strettamente posteriore al train
ANNI_DA_SCARICARE: tuple[int, ...] = (2024, 2025)

TRAIN_START: str = "2024-01-01"
TRAIN_END: str = "2025-09-30"
TEST_START: str = "2025-10-01"
TEST_END: str = "2025-12-31"

DATA_INIZIO: str = TRAIN_START
DATA_FINE: str = TEST_END


# =============================================================================
# Geografia: coordinate di Bologna per l'API meteo storica
# =============================================================================
LAT: float = 44.49
LON: float = 11.34


# =============================================================================
# Selezione spire (15-20 sensori stratificati per funzione urbana)
# =============================================================================
# La traccia (sez. 4) raccomanda 15-25 spire, scelte per copertura urbana
# e non in modo casuale. La lista esatta verrà popolata dopo l'EDA, ma
# fissiamo qui il target dimensionale.
N_SPIRE_TARGET: int = 18


# =============================================================================
# Soglie operative — vedi PIANO.md sez. 3.3 (gestione accuratezza)
# =============================================================================
# Soglia di "alta affidabilità" del sensore: sotto questa, il dato è
# considerato low-confidence e usato in modeling con cautela.
SOGLIA_ACCURATEZZA_AFFIDABILE: float = 80.0

# Soglia di esclusione: sotto questa, il dato non viene proprio usato.
SOGLIA_ACCURATEZZA_MINIMA: float = 50.0

# Soglia per generazione alert: il sistema NON genera alert se l'accuratezza
# è sotto questo livello (per non confondere fenomeno e strumento).
SOGLIA_ACCURATEZZA_ALERT: float = SOGLIA_ACCURATEZZA_AFFIDABILE


# =============================================================================
# Calendario locale di Bologna (festività non gestite dal pkg holidays)
# =============================================================================
# Festività cittadine ricorrenti che impattano il traffico. Da arricchire
# col domain knowledge durante l'EDA. Format: "MM-DD" → etichetta.
FESTIVITA_LOCALI_BOLOGNA: dict[str, str] = {
    "10-04": "San Petronio (patrono)",
}


# =============================================================================
# Parametri modello (default sensati, override nei notebook)
# =============================================================================
# Soglia residuo standardizzato per Baseline 1 (STL+MAD).
# 3.5 è un compromesso tra il classico "3 sigma" e la maggiore conservatività
# del MAD (più robusto). Verrà eventualmente affinato dopo l'EDA.
STL_RESIDUAL_THRESHOLD: float = 3.5

# Contamination per Baseline 2 (IsolationForest).
# 0.02 = ~2% di anomalie attese. Coerente con il framing: stiamo cercando
# eventi rari, non outlier statistici comuni.
ISOLATION_FOREST_CONTAMINATION: float = 0.02

# Numero di alberi per LightGBM / IF.
N_ESTIMATORS: int = 200


# =============================================================================
# Calibrazione soglia ensemble
# =============================================================================
# Percentile sui punteggi del periodo di training per definire la soglia
# operativa del sistema finale. 99 = un alert ogni 100 ore-spira in media.
PERCENTILE_SOGLIA_ENSEMBLE: float = 99.0
