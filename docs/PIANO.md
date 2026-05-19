# Piano dettagliato — Project Work "Predire l'imprevedibile"

> Documento di pianificazione del lavoro, redatto **prima** dello sviluppo come richiesto dalla traccia (sez. 4: "non si parte dalla scelta del modello prima di aver definito il problema"). Contiene struttura del repository, scelte progettuali, finestre temporali, criteri di selezione spire e roadmap esecutiva.

---

## 1. Inquadramento del progetto

La traccia richiede un sistema di **rilevamento anomalie nel traffico cittadino di Bologna** a partire dai dati pubblici delle spire induttive (2019-2025), integrati con dati meteo (Open-Meteo) e calendario festività italiane.

**Vincoli del committente** (volutamente vaghi):
- Segnalare "quando succede qualcosa di insolito nel traffico"
- Distinguere anomalia del fenomeno da malfunzionamento del sensore
- Operare in assenza di etichette di ground truth

**Vincoli del progetto** (espliciti nella traccia):
- Tre approcci di natura diversa (statistica, density-based, forecasting)
- Protocollo di valutazione costruito da zero
- Riproducibilità integrale (seed fissati, dati cached)
- Comunicazione di alta qualità (report tecnico finale)

**Pesi della griglia di valutazione**:
- 30% — Problem framing e coerenza
- 30% — Protocollo di valutazione e analisi critica
- 25% — Solidità tecnica
- 15% — Comunicazione

---

## 2. Struttura del repository (conforme alla traccia, sez. 6.1)

```
06 Predire l'imprevedibile/
├── README.md                      # come riprodurre tutto da zero
├── requirements.txt               # versioni esatte
├── data/
│   ├── raw/                       # parquet originali (spire, accuratezza, meteo)
│   └── processed/                 # dataset master integrato
├── notebooks/
│   ├── 01_eda.ipynb               # Fase 2a — EDA orientata al problema
│   ├── 02_baselines.ipynb         # Fase 2b — Tre baseline
│   ├── 03_sistema_finale.ipynb    # Fase 3a — Composizione ensemble
│   └── 04_valutazione.ipynb       # Fase 3b — Protocollo di valutazione
├── src/                           # codice riutilizzabile (importato dai notebook)
│   ├── __init__.py
│   ├── config.py                  # costanti globali, seed, percorsi
│   ├── data_loading.py            # download + cache spire/meteo/accuratezza
│   ├── preprocessing.py           # melt, merge, feature engineering
│   ├── models.py                  # 3 baseline + ensemble
│   ├── synthetic_anomalies.py     # iniezione anomalie di Fase 3
│   └── evaluation.py              # metriche, protocollo, stabilità
├── results/                       # CSV alert, figure, tabelle finali
└── docs/
    ├── PIANO.md                   # questo documento
    ├── problem_framing.md         # Fase 1 — definizione operativa
    └── report_tecnico.md          # Fase 4 — report finale
```

---

## 3. Scelte progettuali preliminari

### 3.1 Finestra temporale

La traccia consiglia gennaio 2024 → settembre 2025 (~21 mesi). In quel range:
- **2 cicli annuali completi** (necessari per stagionalità annuale)
- Inclusi Ferragosto, Natale, eventi cittadini
- Margine di 3-6 mesi finali per test set strettamente posteriore

**Scelta operativa**: scarichiamo l'intero **2024 + 1° semestre 2025** (18 mesi). Split temporale:
- **Train** : 2024-01-01 → 2025-03-31 (15 mesi)
- **Test**  : 2025-04-01 → 2025-06-30 (3 mesi)

Esclusione esplicita di 2020-2021 (Covid) come da raccomandazione.

### 3.2 Selezione delle 15-20 spire

Criterio di campionamento **stratificato per funzione urbana**, non casuale:
| Categoria | Numero target | Razionale |
|---|---|---|
| Centro storico | 3-4 | dinamica ZTL, eventi |
| Viali (anello) | 3-4 | flusso di scorrimento |
| Tangenziale | 2-3 | traffico di attraversamento |
| Radiali in ingresso | 4-5 | pendolarismo |
| Periferia | 2-3 | regime "domestico" |

**Criterio operativo**:
1. Identifichiamo manualmente strade rappresentative consultando OpenStreetMap
2. Tra le spire di quelle strade selezioniamo quelle con **maggiore copertura** (più giorni con accuratezza > 80% nel periodo)
3. Documenteremo la lista finale in `docs/problem_framing.md`

### 3.3 Gestione dei dati di accuratezza (CRUCIALE)

Coerentemente con la traccia (sez. 3.2: "il cuore del progetto"):
- Soglia di accettazione operativa: **accuratezza ≥ 80%** per fascia oraria
- Ore con accuratezza ∈ [0%, 80%) → flaggate come "low confidence", **escluse dal training**, ma il sistema non genera alert su di esse (vengono presentate al committente come `data_quality_issue`, non come `traffic_anomaly`)
- Ore con accuratezza < 50% → escluse anche dalla validazione

### 3.4 Riproducibilità

- **Seed globale**: `SEED = 42` definito in `src/config.py`
- Fissato in: numpy, sklearn, lightgbm, eventuali sampling
- Tutti i dataset salvati in `data/raw/*.parquet` (cache filesystem)
- `requirements.txt` con pin di versioni esatte (`pip freeze`)

---

## 4. Roadmap esecutiva (mapping fasi traccia ↔ artefatti)

### Fase 1 — Problem framing
**Output**: `docs/problem_framing.md` (2-4 pagine).

**Contenuti**:
1. Definizione operativa di anomalia (focus: **anomalia contestuale** — un'ora che, dato il contesto giorno+ora+meteo+tipo_giorno, è inattesa)
2. Tipologie da catturare / non catturare:
   - **Sì**: drop improvvisi, picchi anomali, plateau, pattern alterati
   - **No**: cali stagionali attesi (Ferragosto, Natale), guasti sensori (gestiti via accuratezza), drift strutturali (deficit di traffico per pandemia/cambio rete viaria)
3. Utente immaginato: **analista del Comune** che produce report settimanali → orienta verso **batch processing**, non real-time; tolleranza maggiore per latenza, intolleranza per FP rumorosi
4. Asimmetria FP/FN: i FP danneggiano molto la credibilità (utente smette di leggere gli alert) → metrica F-beta con β = 0.5 (precisione conta il doppio del recall)

### Fase 2a — EDA
**Output**: `notebooks/01_eda.ipynb`.

**Grafici e analisi**:
1. Profilo orario per giorno settimana (per spira)
2. Stagionalità annuale (mese × ora)
3. Effetto pioggia sul flusso, stratificato per categoria di spira
4. Distribuzione accuratezza (% ore con `acc ≥ 80%`)
5. Eteroschedasticità: std vs ora del giorno (giustifica soglie ora-dipendenti)
6. Visualizzazione di Ferragosto e Natale come "stress-test" attesi

### Fase 2b — Tre baseline
**Output**: `notebooks/02_baselines.ipynb` + `src/models.py`.

| Baseline | Tecnica | Anomalie target | Ipotesi |
|---|---|---|---|
| **1. Statistica** | STL decomposition + MAD su residui | Puntuali (ora isolata estrema) | Stagionalità additiva, residui ~normali (rilassato via MAD) |
| **2. Density-based** | IsolationForest su feature engineered | Contestuali (combinazione anomala di feature) | Anomalie sparse nello spazio feature |
| **3. Forecasting** | LightGBM regressor + score = (y - ŷ) / σ_residui | Contestuali (deviazione da pattern atteso dato il contesto) | Predittabilità del traffico dato lag + meteo + calendario |

**Feature comuni (B2 e B3)**:
- Lag: `lag_1h`, `lag_24h`, `lag_168h`
- Rolling stats su ultime 24 ore (media, std)
- Calendario: `ora`, `giorno_settimana`, `tipo_giorno`, `mese`
- Meteo: `temperature_2m`, `precipitation`, `weather_code` (one-hot grouped)
- Identificativi: `chiave` (categorica)

**Per ciascuna baseline rispondiamo a 3 domande nel notebook**:
1. Che tipi di anomalia cattura naturalmente?
2. Che tipi NON cattura?
3. Su quali ipotesi si basa e sono verificate?

### Fase 3a — Sistema integrato
**Output**: `notebooks/03_sistema_finale.ipynb`.

**Composizione**: **ensemble per massimo dei punteggi normalizzati** (motivato dal framing: vogliamo che basti un approccio per segnalare → recall composito, ma con threshold alto per controllare i FP).

Score finale:
```
score_finale(t,s) = max(score_norm_B1, score_norm_B2, score_norm_B3)
alert(t,s)        = score_finale > soglia ∧ accuratezza ≥ 80%
```

Soglia calibrata sulla **distribuzione del periodo di training** (es. 99° percentile), non per inseguire un target rate fisso.

### Fase 3b — Protocollo di valutazione
**Output**: `notebooks/04_valutazione.ipynb` + `src/evaluation.py`.

**Combiniamo TRE strategie** (la traccia ne chiede almeno 2):

1. **Iniezione di anomalie sintetiche**
   - Tassonomia: spike (+300%), drop (-90%), zero plateau (6 ore consecutive), shift (+50% costante per 12 ore)
   - Anomalie "borderline" (non solo +500%) per evitare il problema della traccia (sez. 8): anomalie troppo facili → falsa sicurezza
   - 100 iniezioni random nel test set, replicato 5 volte con seed diversi (per stabilità)
   - Metrica: **detection rate per tipologia** (recall stratificato) + **mean lead time**

2. **Validazione con eventi reali**
   - Lista manuale di 15-20 eventi 2024-2025 (es. partite Bologna FC al Dall'Ara, sciopero TPL del 12/04/2024, neve dell'8/01/2024, Salone del Mobile/BolognaFiere)
   - Per ogni evento: ci aspettiamo alert in finestra ±3 ore su spire georeferenziate vicine
   - Metrica: **hit rate eventi** (eventi catturati / eventi totali)

3. **Analisi di stabilità**
   - Variazione di: soglia (90°→99° percentile), contamination IF (0.01→0.05), window size lag
   - Misurare **Jaccard index** tra insiemi di alert al variare degli iperparametri
   - Metrica: **stabilità J(α₁, α₂)** — più alta = più robusto

**Metriche custom motivate**:
- **F-0.5 score** su anomalie sintetiche (β < 1 per penalizzare FP, coerente con framing)
- **AUC-PR** sulle iniezioni sintetiche
- **Operational alert rate** = % ore-spira con alert / ore-spira totali (deve essere "leggibile" da analista: target <0.5%)

### Fase 4 — Report
**Output**: `docs/report_tecnico.md`.

Sezioni (come da traccia sez. 6.2):
1. Introduzione e contesto
2. Problem framing (raffinato)
3. Dati e preprocessing
4. Esplorazione (solo evidenze rilevanti)
5. Modelli
6. Valutazione
7. **Limiti e discussione critica** (sezione obbligatoria, valutata pesantemente)
8. Conclusioni e lavori futuri
9. Bibliografia

---

## 5. Strategia "build incrementale" per gestire la complessità

Dato che il progetto è pluri-settimanale, lavoreremo a moduli con commit atomici:

| Step | Output minimo | Verifica |
|---|---|---|
| 1 | `src/config.py` + `src/data_loading.py` con download cache | dati scaricati in `data/raw/*.parquet` |
| 2 | `src/preprocessing.py` con master dataset integrato | `data/processed/master.parquet` ispezionabile |
| 3 | `notebooks/01_eda.ipynb` eseguito | grafici salvati in `results/figures/` |
| 4 | `src/models.py` con 3 baseline implementate (e testate su mini-dataset) | smoke test in fondo a `models.py` |
| 5 | `notebooks/02_baselines.ipynb` completo | tabelle riassuntive in `results/` |
| 6 | `src/synthetic_anomalies.py` | iniezione testata su subset |
| 7 | `notebooks/03_sistema_finale.ipynb` | CSV di alert in `results/alerts.csv` |
| 8 | `src/evaluation.py` + `notebooks/04_valutazione.ipynb` | risultati tabella metriche |
| 9 | `docs/problem_framing.md` + `docs/report_tecnico.md` | documenti finali |
| 10 | `README.md` + `requirements.txt` | repository riproducibile |

---

## 6. Rischi noti e mitigazioni

| Rischio | Probabilità | Mitigazione |
|---|---|---|
| Download dataset 18 mesi lento / fallisce per timeout | media | Chunking annuale + retry exponential backoff |
| Memoria insufficiente con 20 spire × 18 mesi (~262k righe) | bassa | Parquet + dtype downcast (int16 per conteggi) |
| Confusione anomalia fenomeno/sensore | alta | Soglia accuratezza 80% blindata in pipeline + flag esplicito nel dataset |
| Test set troppo facile (anomalie sintetiche enormi) | media | Tassonomia con anomalie borderline + cross-check stabilità |
| Fuso orario UTC vs Europa/Roma nei meteo | alta | `timezone=Europe/Rome` esplicito (vedi notebook demo) + unit test su join |
| Sovra-engineering del codice (`src/`) | media | Smoke test in fondo a ogni modulo + import-only dai notebook |

---

## 7. Note didattiche (parte teorica da inserire nei notebook)

Ogni notebook avrà sezioni teoriche per spiegare i concetti agli studenti:

**Notebook 1 (EDA)**:
- Perché serve l'EDA orientata al problema (non l'EDA "di stile")
- Stagionalità multipla (concetto)
- Eteroschedasticità e perché soglie statiche sono sbagliate

**Notebook 2 (baseline)**:
- STL decomposition: trend, stagionalità, residuo (teoria + intuizione visiva)
- MAD vs std: perché il MAD è robusto (formula + esempio numerico)
- IsolationForest: idea dell'isolamento, profondità nell'albero come score
- Forecasting-based detection: il "residuo standardizzato" come score di anomalia

**Notebook 3 (sistema finale)**:
- Ensemble di score: media vs max vs voto (trade-off teorici)
- Calibrazione di score eterogenei (z-score, rank, min-max)

**Notebook 4 (valutazione)**:
- Il problema della valutazione senza ground truth (paradosso anomaly detection)
- Anomalie sintetiche: tipologie e perché borderline > estreme
- Jaccard index come misura di stabilità

---

## 8. Definition of Done

Il progetto è "fatto" quando:
- [x] Tutti gli artefatti elencati in sez. 4 esistono
- [x] `pip install -r requirements.txt && jupyter nbconvert --execute notebooks/*.ipynb` rieseque tutto senza errori (in un ambiente pulito)
- [x] Il report copre tutte le sezioni richieste, con particolare cura per "Limiti e discussione critica"
- [x] I seed sono fissati ovunque (verificabile con grep)
- [x] Il README permette a un terzo di clonare e rieseguire tutto

---

## Prossimo step

Procediamo con lo **Step 1** della roadmap: `src/config.py` + `src/data_loading.py` per scaricare i dati 2024+2025 in cache locale. Il resto procederà sequenzialmente.
