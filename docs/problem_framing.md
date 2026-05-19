# Problem Framing Document — Fase 1

> **Output della Fase 1** della traccia (vedi PIANO.md sez. 4).
> Documento vincolante per tutte le scelte tecniche successive. Eventuali
> deviazioni rispetto a quanto qui dichiarato dovranno essere esplicitate e
> motivate nel report tecnico finale.

---

## 1. Contesto

L'amministrazione del Comune di Bologna dispone di una rete di **spire induttive** (sensori sotto l'asfalto che contano i veicoli) e pubblica i dati orari di flusso sul portale Open Data. La richiesta del committente è volutamente sfuocata:

> *"Vorremmo capire quando succede qualcosa di insolito nel traffico cittadino. Non sappiamo esattamente cosa cercare, ma sappiamo che ci sono giorni e ore in cui i flussi si comportano in modo che non ci aspettiamo, e vorremmo un sistema che ce li segnali."*

Il nostro lavoro è trasformare questa richiesta in una **definizione operativa di anomalia**, un sistema di rilevamento, e un protocollo di valutazione costruito in assenza di etichette ground-truth.

## 2. Definizione operativa di anomalia

> **Un'anomalia è un'osservazione `(spira, ora)` il cui conteggio veicoli si discosta significativamente dal valore atteso, dato il contesto temporale (ora del giorno, giorno della settimana, tipo di giorno), meteorologico, e storico recente della spira stessa. Devono essere catturate sia singole ore isolate (anomalie puntuali contestuali) sia sequenze di ore consecutive con un pattern alterato (anomalie collettive contestuali).**

In linguaggio operazionale, possiamo riformulare così:
- L'unità di osservazione è la coppia `(chiave_spira, timestamp_orario)`.
- Per ciascuna unità, il sistema produce uno **score di anomalia continuo** (più alto = più anomalo).
- Un'osservazione diventa **alert** se lo score supera una soglia operativa calibrata e se il dato è affidabile (accuratezza ≥ 80%).

### 2.1 Tipologia (rispetto alla tassonomia letteratura)

Seguendo Chandola, Banerjee & Kumar (2009), distinguiamo:
- **Anomalie puntuali**: un singolo dato fuori scala (es. 5000 veicoli/h in una via dove la media è 200).
- **Anomalie contestuali**: un dato normale in assoluto ma anomalo in quel contesto (es. 200 veicoli/h alle 8 del mattino in via Indipendenza — fisiologicamente è "poco" per quel contesto, anche se 200 in assoluto è banale).
- **Anomalie collettive**: una sequenza di osservazioni che insieme formano un pattern strano (es. 6 ore consecutive a zero in un giorno feriale).

**Il nostro focus primario è sulle anomalie contestuali**. Le puntuali estreme sono catturate "gratuitamente" come caso degenere; le collettive vengono rilevate come ammasso di anomalie contestuali consecutive, segnalate nel report finale come "evento sospetto" con durata.

## 3. Cosa il sistema DEVE catturare

1. **Drop improvvisi del flusso** in ore "vive" (8-20) su strade abitualmente trafficate
   - Esempi: blocco stradale per incidente, sciopero TPL che blocca un asse viario
2. **Picchi anomali** non spiegabili da meteo o calendario
   - Esempi: manifestazione, evento sportivo, fiera in zona
3. **Plateau** (es. 6+ ore consecutive a valori molto diversi dall'atteso)
   - Esempi: chiusura prolungata di una strada per lavori d'emergenza
4. **Inversione di pattern**: una via che mostra il profilo di una via di tipologia diversa
   - Esempi: una via centrale che mostra pattern da periferia per via di un blocco strutturale

## 4. Cosa il sistema NON DEVE catturare

1. **Cali stagionali attesi**
   - Ferragosto, Natale, finesettimana lunghi: il calo c'è ma è noto e fisiologico → il modello "sa" che è atteso perché incorporiamo `tipo_giorno` e mese come feature.
2. **Variazioni meteo "normali"**
   - Pioggia moderata che riduce il traffico del 10% → atteso, non è un'anomalia.
3. **Guasti del sensore** (accuratezza < 80%)
   - Vengono filtrati a monte e gestiti separatamente come `data_quality_issue`, NON come `traffic_anomaly`. Il committente vuole sapere del traffico, non dei sensori.
4. **Drift strutturali pluriennali**
   - Cambi di assetto viario, ZTL modificate: queste deviazioni sono sostenute e graduali, vengono "imparate" come nuova normalità una volta che il modello viene riaddestrato. Non sono anomalie nel senso operativo.

## 5. Utente finale

> **Profilo target: analista del settore Mobilità del Comune di Bologna che produce report settimanali sui pattern di traffico.**

Implicazioni progettuali:

| Dimensione | Scelta operativa | Motivazione |
|---|---|---|
| Latenza | Batch giornaliero/settimanale | Non serve real-time per un report |
| Granularità output | Lista di ore-spira anomale + cluster temporali | Analista vuole capire il "cosa", non solo il "dove" |
| Volume tollerato di alert | Ordine 10-50 alert/settimana | Oltre l'analista smette di leggere |
| Spiegabilità | Score + 3 feature più influenti | Aiuta l'analista a interpretare e validare |

**Esclusi esplicitamente da questo design**:
- Dispatcher di polizia municipale (richiederebbe real-time)
- Sistema di controllo semaforico adattivo (richiederebbe latenza < secondo)

## 6. Asimmetria del costo degli errori

**Costo di un falso positivo** (alert su un'ora normale):
- L'analista perde tempo investigando un non-evento
- Erosione di fiducia nel sistema (se gli alert sono spesso falsi, smettono di essere letti)
- Effetto "cry-wolf" cumulativo

**Costo di un falso negativo** (anomalia non segnalata):
- Un evento sfugge al monitoring settimanale
- Recuperabile a posteriori incrociando con cronaca, dati del Comune, social
- Singolo evento perso è meno grave di una valanga di FP

**Verdetto**: **i falsi positivi sono più costosi dei falsi negativi**. Operativamente:
- Useremo **F-0.5 score** (β=0.5: la precisione pesa il doppio del recall) come metrica composita principale
- La calibrazione della soglia ensemble punterà al **99° percentile** dei punteggi di training, per generare alert sparsi e ad alta confidenza
- Nel report finale presenteremo precision e recall separati, non aggregati

## 7. Limiti accettati in fase di framing

Riconosciamo esplicitamente che il sistema risultante:
1. **Non distinguerà la causa** dell'anomalia (incidente vs manifestazione vs sciopero) — segnalerà solo che il pattern è anomalo
2. **Funzionerà solo sulle 18 spire selezionate**, non sull'intera rete cittadina
3. **Dipenderà dalla qualità del dataset di accuratezza** — eventuali bias sistematici in quel dato si propagano
4. **Avrà un periodo di "warm-up"** all'inizio del training: i lag feature richiedono almeno una settimana di dati storici per essere disponibili
5. **Sarà cieco ad anomalie che durano tutta la finestra di training** (drift uniforme passato come normalità)

Questi limiti saranno discussi in dettaglio nella sezione obbligatoria "Limiti e discussione critica" del report finale.

## 8. Roadmap concordata

La definizione di anomalia sopra implica le seguenti scelte tecniche, già riflesse in `PIANO.md`:
- **Tre baseline complementari**, ciascuna sensibile a una sotto-categoria di anomalia
- **Ensemble per massimo dei punteggi normalizzati** (basta un approccio per segnalare → recall composito)
- **Calibrazione soglia su percentile training** (non target rate fisso)
- **Filtro accuratezza ≥ 80%** prima della generazione degli alert
- **Valutazione con 3 strategie**: iniezione sintetica, eventi reali, stabilità iperparametri

## 9. Riferimenti

- Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly detection: A survey*. ACM Computing Surveys.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). *STL: A Seasonal-Trend Decomposition Procedure Based on Loess*. J. of Official Statistics.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. ICDM.
- Hyndman, R. J. & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.), Cap. 3 ("Decomposition") e 13 ("Hierarchical and grouped time series").
