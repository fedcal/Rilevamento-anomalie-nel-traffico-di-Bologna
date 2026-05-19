# Predire l'imprevedibile — Rilevamento anomalie nel traffico di Bologna

> Project work del corso DataMasters/Skiller "Machine Learning Engineer".
> Obiettivo: costruire un sistema di **anomaly detection** sui flussi orari delle spire induttive del Comune di Bologna (open data 2024-2025), integrati con dati meteo e calendario festività, e validarlo in assenza di etichette di ground truth.

## Struttura del repository

```
.
├── README.md                          # questo file
├── requirements.txt                   # dipendenze Python
├── 06 Predire l'imprevedibile - ...md # traccia originale del project work
│
├── data/
│   ├── raw/                           # dati grezzi scaricati (Parquet, ~40 MB)
│   │   ├── flussi_2024.parquet
│   │   ├── flussi_2025.parquet
│   │   ├── accuratezza_2024.parquet
│   │   ├── accuratezza_2025.parquet
│   │   └── meteo_2024-01-01_2025-06-30.parquet
│   └── processed/
│       ├── master.parquet             # dataset analitico integrato
│       ├── baselines_scores.parquet   # score delle 3 baseline
│       └── ensemble_scores.parquet    # score finale
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Fase 2.a — EDA orientata al problema
│   ├── 02_baselines.ipynb             # Fase 2.b — Tre baseline
│   ├── 03_sistema_finale.ipynb        # Fase 3.a — Ensemble
│   └── 04_valutazione.ipynb           # Fase 3.b — Protocollo di valutazione
│
├── src/                               # codice riutilizzabile
│   ├── config.py                      # costanti, seed, percorsi
│   ├── data_loading.py                # download Bologna + Open-Meteo
│   ├── preprocessing.py               # melt, merge, feature engineering
│   ├── models.py                      # 3 baseline + ensemble
│   ├── synthetic_anomalies.py         # iniezione anomalie
│   └── evaluation.py                  # metriche e protocollo
│
├── results/                           # output del progetto
│   ├── figures/                       # grafici dei notebook
│   ├── alerts_finali.csv              # deliverable principale per il committente
│   ├── alerts_clusters_test.csv       # alert raggruppati in eventi
│   ├── ablation_baseline.csv          # confronto B1/B2/B3 vs ensemble
│   ├── matrice_stabilita.csv          # Jaccard al variare del percentile
│   ├── validazione_eventi_reali.csv   # match con 18 eventi storici
│   ├── metriche_sintetiche.json       # metriche su anomalie iniettate
│   └── sintesi_valutazione.json       # dashboard finale
│
└── docs/
    ├── PIANO.md                       # piano di lavoro (Fase 0)
    ├── problem_framing.md             # Fase 1 — definizione operativa
    └── report_tecnico.md              # Fase 4 — report finale
```

## Requisiti

- Python ≥ 3.10 (testato su 3.13)
- ~500 MB di spazio disco (dati grezzi + cache)
- Connessione internet per il primo download (~40 MB)

## Come riprodurre tutto da zero

```bash
# 1. Clona il repository e entra nella directory del project work
cd "06 Predire l'imprevedibile - Rilevamento anomalie nel traffico di Bologna"

# 2. Crea un ambiente virtuale e installa le dipendenze
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# 3. Scarica i dati grezzi (~5 min, dipende dall'API)
./venv/bin/python -m src.data_loading

# 4. Costruisci il master dataset (1-2 min)
./venv/bin/python -m src.preprocessing

# 5. Esegui i notebook nell'ordine (15-20 min totali)
./venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
./venv/bin/jupyter nbconvert --to notebook --execute notebooks/02_baselines.ipynb --output 02_baselines.ipynb
./venv/bin/jupyter nbconvert --to notebook --execute notebooks/03_sistema_finale.ipynb --output 03_sistema_finale.ipynb
./venv/bin/jupyter nbconvert --to notebook --execute notebooks/04_valutazione.ipynb --output 04_valutazione.ipynb
```

## Riproducibilità

Tutti i seed sono fissati a `42` in `src/config.py:SEED`. Le librerie con RNG proprio (sklearn, lightgbm) ricevono `random_state=SEED` esplicito al costruttore.

## Lettura consigliata

Per chi vuole capire il progetto:

1. **`docs/problem_framing.md`** (~5 min) — l'inquadramento del problema
2. **`docs/PIANO.md`** (~10 min) — il piano di lavoro
3. **`notebooks/01_eda.ipynb`** (~15 min) — i dati visualizzati
4. **`notebooks/02_baselines.ipynb`** (~30 min) — le tre baseline con teoria
5. **`notebooks/03_sistema_finale.ipynb`** (~10 min) — composizione ensemble
6. **`notebooks/04_valutazione.ipynb`** (~30 min) — il protocollo di validazione
7. **`docs/report_tecnico.md`** (~20 min) — sintesi e discussione critica

## Deliverable principale

Il **CSV `results/alerts_finali.csv`** è il prodotto consegnabile al committente: una riga per ciascun alert generato, con timestamp, identificativo spira, valore osservato, score di anomalia, e contesto (meteo, tipo giorno).

## Limiti dichiarati

Vedi `docs/problem_framing.md` (sez. 7) e `docs/report_tecnico.md` (sezione "Limiti e discussione critica") per la lista completa delle assunzioni e dei limiti del sistema.

## Licenza dei dati

I dati di flusso e accuratezza sono open data del Comune di Bologna ([dati.comune.bologna.it](https://dati.comune.bologna.it/dati/flusso-veicolare)). I dati meteo provengono da [Open-Meteo](https://open-meteo.com/). Entrambe le fonti sono utilizzabili per scopi non commerciali con attribuzione.
