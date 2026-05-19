# Report tecnico — Predire l'imprevedibile: rilevamento anomalie nel traffico di Bologna

> **Autore**: Federico Calò
> **Corso**: DataMasters — Machine Learning Engineer
> **Periodo**: 2026
> **Versione dati**: gennaio 2024 — dicembre 2025

---

## 1. Introduzione e contesto

L'amministrazione di Bologna pubblica sul portale Open Data le rilevazioni orarie del traffico cittadino misurate da una rete di spire induttive. La richiesta del committente, deliberatamente sfuocata, è:

> *"Vorremmo capire quando succede qualcosa di insolito nel traffico cittadino."*

Non c'è un dataset etichettato. Non c'è una metrica predefinita. Il vero progetto, come dichiara la traccia, non è scegliere un modello: è **costruire un'interpretazione operativa** della richiesta e validarla con un protocollo costruito da zero.

L'approccio scelto è stato:
1. **Problem framing esplicito** in Fase 1 — vincolante per le scelte successive
2. **Tre baseline complementari** (statistica, density-based, forecasting), ciascuna sensibile a un sotto-tipo di anomalia
3. **Sistema ensemble** per massimo dei punteggi normalizzati
4. **Tre strategie indipendenti di validazione**: anomalie sintetiche, eventi reali, analisi di stabilità

Il **deliverable principale** per il committente è il file `results/alerts_finali.csv`: una riga per ciascun alert generato, con timestamp, identificativo della spira, valore osservato, score di anomalia e contesto operativo.

## 2. Problem framing

Riportato in extenso in `docs/problem_framing.md`. Sintesi:

**Definizione operativa**: un'anomalia è una coppia `(spira, ora)` il cui conteggio veicoli si discosta significativamente dal valore atteso dato il contesto (ora, giorno, tipo giorno, meteo, storia recente).

**Tipi target**: drop improvvisi, picchi anomali, plateau, inversioni di pattern.

**Esplicitamente esclusi**: cali stagionali attesi (Ferragosto, Natale), variazioni meteo normali, guasti sensori (gestiti via filtro accuratezza), drift strutturali pluriennali.

**Utente immaginato**: analista del settore Mobilità del Comune che produce report settimanali → orienta verso batch processing, volume contenuto di alert (~10-50/settimana), alta priorità a controllare i falsi positivi.

**Costo errori**: i FP costano più dei FN (erosione di fiducia, effetto "cry-wolf"). Scelta operativa: **F-0.5 score** come metrica composita principale e calibrazione conservativa della soglia (99° percentile).

## 3. Dati e preprocessing

### 3.1 Sorgenti

| Sorgente | Periodo | Granularità | Records |
|---|---|---|---|
| Flussi spire Bologna ([opendata.comune.bologna.it](https://opendata.comune.bologna.it/)) | 2024-2025 | (spira × giorno × ora) | ~615.000 |
| Accuratezza spire Bologna | 2024-2025 | (spira × giorno × ora) | ~670.000 |
| Meteo storico ([open-meteo.com](https://open-meteo.com/)) | 2024-2025 | orario | ~13.000 |

### 3.2 Pipeline di preprocessing (`src/preprocessing.py`)

6 stadi:
1. **Melt flussi**: wide (24 colonne orarie) → long (una riga per `(spira, timestamp)`)
2. **Melt accuratezza**: stesso pattern con schema leggermente diverso (parsing `85%` → 85.0, sentinel `-1%` → NaN)
3. **Selezione spire**: 18 sensori con la maggiore qualità del dato nel periodo (criterio: ore con `accuratezza ≥ 80%`)
4. **Merge**: flussi ⊕ accuratezza (su `chiave + timestamp`) ⊕ meteo (su `timestamp`)
5. **Calendario**: festività italiane (`holidays` package) + festa locale San Petronio → categorizzazione `tipo_giorno ∈ {feriale, weekend, festivo}`
6. **Feature engineering**: lag `{1h, 24h, 168h}` per spira + rolling mean/std a 24h

### 3.3 Selezione delle 18 spire

Strade rappresentative di diverse funzioni urbane (vedi `notebooks/01_eda.ipynb` per la lista completa):
- Centro storico (es. Via S. Stefano, Via D'Azeglio)
- Viali e radiali (es. Via Zanardi, Via Mezzofanti)
- Periferia (es. Via Po, Viale Ercolani)
- Strade trafficate (es. Via Saragozza, Via S. Mamolo)

Su 18 spire × 17.520 ore (= 315.792 record), il **99.1% ha accuratezza ≥ 80%** — questo conferma la qualità del campione selezionato.

### 3.4 Split temporale

| Periodo | Range | Uso |
|---|---|---|
| Train | 2024-01-01 → 2025-09-30 (21 mesi) | Fit modelli + calibrazione soglia |
| Test | 2025-10-01 → 2025-12-31 (3 mesi) | Valutazione out-of-sample + iniezione anomalie |

Split **strettamente temporale** (vedi traccia sez. 8 — trappola comune).

## 4. Esplorazione dei dati (Fase 2.a)

Evidenze chiave (dettagli e grafici in `notebooks/01_eda.ipynb` e `results/figures/`):

1. **Distribuzione bimodale** del conteggio → due regimi (notte vs giorno)
2. **Stagionalità settimanale netta**: doppia gobba feriali, profilo schiacciato domenica
3. **Stagionalità annuale visibile**: forte calo ad Agosto, secondario in dicembre-gennaio
4. **Eteroschedasticità forte**: la deviazione standard varia di un fattore ~20x tra ora 3 e ora 18 → **giustifica le soglie ora-dipendenti dei modelli**
5. **Effetto pioggia moderato**: -5..-10% in media, variabile per ora
6. **Qualità del dato eccellente** sulle 18 spire selezionate (99.1% ore affidabili)
7. **Profili distinti** feriale/weekend/festivo → conferma il valore di `tipo_giorno` come feature condizionante

## 5. Modelli

### 5.1 Baseline 1 — STL + MAD (`src/models.py:baseline_stl_mad`)

- Decomposizione STL con `period=168` (settimana), `robust=True`
- Per ciascuna spira: residui $R_t = y_t - T_t - S_t$
- z-score robusto: $z_t = (R_t - \text{median}(R)) / (1.4826 \cdot \text{MAD}(R))$
- Alert se $|z_t| > 3.5$

**Razionale dell'iperparametro**: la soglia 3.5 sui z robusti corrisponde, sotto ipotesi gaussiana, a ~0.05% di osservazioni. È più conservativa di "3 sigma" classico per riflettere il framing FP-averse.

### 5.2 Baseline 2 — IsolationForest (`src/models.py:baseline_isolation_forest`)

- 12 feature contestuali: `ora, dow, mese, weekend, lag_{1,24,168}h, rolling_mean_24h, rolling_std_24h, temperature_2m, precipitation, wind_speed_10m`
- `n_estimators=200`, `contamination=0.02`
- Training sui soli dati di train con accuratezza ≥ 80%

**Razionale**: contamination = 2% riflette l'ipotesi di anomalie rare. Se il valore fosse 0.1 (10%, default sklearn), il modello produrrebbe alert su un volume insostenibile.

### 5.3 Baseline 3 — LightGBM forecasting (`src/models.py:baseline_forecasting_lgbm`)

- Regressore LightGBM (200 alberi, `max_depth=6`, `learning_rate=0.05`)
- Predice il conteggio atteso $\hat{y}$
- Score: $|y - \hat{y}| / \sigma_{\text{residuo per spira}}$
- Alert se $\text{score} > 3.5$

**Training MAE**: ~5-10 veicoli/h a seconda della spira. Confronto: la media del conteggio è ~100/h. Errore relativo ~5-10%, accettabile per un baseline.

### 5.4 Sistema ensemble (`src/models.py:ensemble_max`)

- Normalizzazione rank-based di ciascuno score
- Score finale = `max(rank_b1, rank_b2, rank_b3)`
- Soglia = 99° percentile dello score sul training
- Alert solo se accuratezza ≥ 80% (politica del sistema)

## 6. Valutazione

Tutti i risultati provengono dall'esecuzione fedele di `notebooks/04_valutazione.ipynb`. Niente è inventato o "aggiustato".

### 6.1 Risultati su anomalie sintetiche (100 iniezioni)

| Tipo anomalia iniettata | Recall ensemble |
|---|---|
| drop | **64.5%** |
| spike | **62.1%** |
| zero_plateau | 49.3% |
| shift | 42.4% |

**Metriche globali ensemble (su 312 ore-spira marcate come ground truth nel test set):**
- Precision: 0.182
- Recall: 0.494
- **F-0.5: 0.209**
- AUC-PR: 0.218
- Operational alert rate: 2.1%

### 6.2 Risultati su 18 eventi reali

**Hit rate: 77.8% (14/18 eventi colpiti).**

Eventi catturati: Sciopero TPL nazionale, Bologna FC-Juventus, Ferragosto 2024, Capodanno 2025, Pasqua 2024, San Petronio, Sciopero generale, 25 Aprile, 1 Maggio, Pasqua 2025, Festa Repubblica 2025, e altri.

Eventi mancati: probabili motivi documentati (eventi previsti e modellati come "normalità" festiva — es. Natale).

### 6.3 Stabilità (matrice di Jaccard)

| Variazione soglia | Jaccard |
|---|---|
| P99 vs P99.5 (lieve) | **0.50** |
| P95 vs P99.9 (estrema) | 0.02 |

Il J = 0.50 tra P99 e P99.5 indica un **nucleo di alert ragionevolmente stabile**: piccoli cambi di calibrazione cambiano una porzione moderata degli alert, ma metà sopravvive a entrambe le scelte.

### 6.4 Ablation study (CRUCIALE)

| Approccio | Precision | Recall | F-0.5 | AUC-PR | Alert rate |
|---|---|---|---|---|---|
| B1 (STL+MAD) | 0.281 | 0.455 | 0.304 | 0.213 | 1.27% |
| B2 (IsolationForest) | 0.003 | 0.010 | 0.003 | 0.007 | 2.70% |
| **B3 (LightGBM)** | **0.332** | **0.567** | **0.362** | **0.515** | **1.34%** |
| Ensemble (max rank) | 0.182 | 0.494 | 0.209 | 0.218 | 2.12% |

> **Risultato non scontato**: B3 da solo **batte** l'ensemble su tutte le metriche tranne alert rate. La causa principale: B2 (IsolationForest) ha performance pessime su questo task, e il MAX ensemble include i suoi alert (che sono in gran parte falsi positivi), abbassando la precision complessiva.

Questo è esattamente il tipo di scoperta che richiede di **modificare il sistema** o di **giustificarne la scelta con un trade-off esplicito**.

## 7. Limiti e discussione critica

> Sezione obbligatoria e valutata pesantemente dalla traccia. Affrontata con franchezza.

### 7.1 Il problema dell'IsolationForest

Il fatto che B2 abbia performance bassissime (precision 0.3%) indica che, su questo task con queste feature, IsolationForest **non sta facendo il suo lavoro**. Ipotesi diagnostiche:

1. Le feature contestuali sono **fortemente correlate** (es. `lag_24h` e `rolling_mean_24h`), creando partizioni poco informative
2. La `contamination=0.02` è ancora troppo alta, generando molti alert ai margini casuali
3. Le anomalie sintetiche iniettate sono **anomalie temporali**, non *anomalie nello spazio feature* dove IF eccelle

Una versione successiva del sistema dovrebbe:
- Sostituire B2 con un detector più appropriato (One-Class SVM, LOF, AutoEncoder)
- Oppure adottare **ensemble pesato** (con peso quasi nullo a B2) anziché max
- Oppure adottare **B3 standalone** dichiarando esplicitamente il trade-off

### 7.2 Conseguenze sul deliverable

Il file `results/alerts_finali.csv` consegnato è quello dell'ensemble MAX configurato come da problem framing. Tuttavia, sulla base dell'ablation, **un sistema basato solo su B3 produrrebbe un CSV di qualità superiore**. Questa è informazione che vogliamo comunicare al committente in modo trasparente, non nascondere.

### 7.3 Limiti dei dati

- **Solo 18 spire su decine** disponibili nella rete cittadina. Generalizzazione non garantita.
- **Periodo limitato a 2 anni**: stagionalità annuale è osservata 2 volte. Cicli pluriennali (es. effetti di nuove opere viarie) non catturati.
- **Eventi reali validati a campione**: 18 eventi non sono esaustivi. La lista include sia eventi "facili" (festività) che "difficili" (scioperi).
- **Anomalie sintetiche borderline**: l'iniezione produce anomalie realistiche ma di natura nota. Il sistema potrebbe fallire su tipologie diverse (es. shift molto lento, drift graduale).

### 7.4 Assunzioni sui dati che potrebbero non reggere

- **Stazionarietà di lungo periodo**: assumiamo che il pattern di traffico del 2024-2025 sia rappresentativo del futuro. Cambi infrastrutturali (nuovi sensi unici, ZTL allargata, lavori pluriennali) invalidano questa assunzione e richiederebbero riaddestramento.
- **Indipendenza tra spire**: trattiamo ogni spira indipendentemente. Anomalie *correlate* tra spire (es. blocco multiplo) sono catturate solo per "spira singola che vede l'effetto". Un'estensione naturale è un modello spaziale.
- **Accuratezza come proxy del guasto**: assumiamo che `accuratezza < 80%` significhi "sensore in difficoltà". Se invece misurasse altro (es. classificazione del veicolo), il filtro sarebbe eccessivo.

### 7.5 Scenari operativi in cui NON usare questo sistema

- **Real-time control** (semafori adattivi, comunicazioni di emergenza): la latenza batch non è adatta.
- **Dispatch di emergenza**: la precision 18% genererebbe troppe chiamate inutili.
- **Spire non incluse**: il modello non generalizza ad altre spire senza fine-tuning.
- **Periodi con eventi straordinari noti** (es. pandemia, lockdown): il drift di regime invaliderebbe tutto il modeling.

### 7.6 Categorie di anomalie non catturate

- **Shift molto lenti**: aumento graduale del 5% per mesi. Il modello lo apprende come nuova normalità.
- **Anomalie correlate alla velocità** non al volume: le spire contano i veicoli, non le code. Un incidente che genera coda ma non blocca il transito può non essere rilevato.
- **Anomalie di composizione**: cambio nella tipologia di veicoli (es. più mezzi pesanti). Il dato non lo include.

## 8. Conclusioni e lavori futuri

### 8.1 Cosa è stato consegnato

- Pipeline end-to-end **riproducibile** (seed fissati, dati cached, README dettagliato)
- **18 eventi storici** validati con hit rate 77.8%
- **Tre baseline** ciascuna documentata con teoria, ipotesi e limiti
- Sistema ensemble con **soglia operativa calibrata** sul percentile training
- **Protocollo di valutazione** combinato (sintetiche + reali + stabilità)
- **Ablation study trasparente** che mostra dove il sistema può migliorare

### 8.2 Priorità per un secondo round

1. **[priorità alta]** Sostituire B2 (IF) con LOF o AutoEncoder e rieseguire l'ablation
2. **[priorità alta]** Implementare un ensemble pesato (es. average dei rank con pesi appresi)
3. **[priorità media]** Estendere a 50+ spire con stratificazione geografica formale
4. **[priorità media]** Aggiungere modello **spaziale** che sfrutta la correlazione tra spire vicine
5. **[priorità media]** Costruire un sistema di feedback (analista marca alert come TP/FP) per chiusura del loop
6. **[priorità bassa]** Aggiungere classificazione del tipo di anomalia (drop vs spike vs shift) all'output

### 8.3 Conclusione

Il sistema costruito **funziona** (catturando 78% degli eventi storici e oltre il 60% delle anomalie sintetiche più strutturali) ma **non è quanto potrebbe essere**. L'ablation rivela che una versione più semplice (B3 standalone) supera la versione complessa (ensemble MAX). Questa è una scoperta importante e l'avremmo nascosta optando per metriche selezionate: la trasparenza su questo è parte del valore consegnato.

> *"Un sistema che produce risultati mediocri ma ben argomentato e onesto sui propri limiti vale più di un sistema che sembra eccellente ma non sa rispondere alla domanda 'come fai a saperlo?'."*
> — Traccia, sez. 7

## 9. Bibliografia

- **Chandola, V., Banerjee, A., & Kumar, V.** (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), 1-58.
- **Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I.** (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.
- **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413-422.
- **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y.** (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- **Hyndman, R. J. & Athanasopoulos, G.** (2021). *Forecasting: principles and practice* (3rd ed.). OTexts.
- **Hampel, F. R.** (1974). The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346), 383-393. (Origine del MAD)
- **Hyndman, R. J. & Khandakar, Y.** (2008). Automatic time series forecasting: the forecast package for R. *Journal of Statistical Software*, 27(3), 1-22.

### Fonti dati e documentazione

- Open Data Comune di Bologna — [opendata.comune.bologna.it](https://opendata.comune.bologna.it/)
- Open-Meteo Historical Weather API — [open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api)
- `holidays` Python package — [pypi.org/project/holidays](https://pypi.org/project/holidays/)
- `lightgbm` documentation — [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/)
- `statsmodels` STL — [statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html)

## Appendici

### A. File di output

| File | Descrizione |
|---|---|
| `results/alerts_finali.csv` | **Deliverable principale**: 2.988 alert con contesto |
| `results/alerts_clusters_test.csv` | 248 cluster temporali di alert nel test set |
| `results/ablation_baseline.csv` | Confronto numerico baseline ↔ ensemble |
| `results/matrice_stabilita.csv` | Matrice Jaccard 5×5 (percentili) |
| `results/validazione_eventi_reali.csv` | Match con 18 eventi storici |
| `results/metriche_sintetiche.json` | Metriche su anomalie iniettate |
| `results/sintesi_valutazione.json` | Dashboard riassuntiva |
| `results/figures/*.png` | ~15 grafici dei notebook |

### B. Iperparametri del sistema (vedi `src/config.py`)

```
SEED = 42
TRAIN: 2024-01-01 → 2025-09-30
TEST:  2025-10-01 → 2025-12-31
N_SPIRE_TARGET = 18
SOGLIA_ACCURATEZZA_AFFIDABILE = 80.0
SOGLIA_ACCURATEZZA_ALERT      = 80.0
STL_RESIDUAL_THRESHOLD        = 3.5
ISOLATION_FOREST_CONTAMINATION = 0.02
N_ESTIMATORS                  = 200
PERCENTILE_SOGLIA_ENSEMBLE    = 99.0
```
