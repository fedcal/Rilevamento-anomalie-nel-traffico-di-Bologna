# 06 Predire l'imprevedibile - Rilevamento anomalie nel traffico di Bologna

Project work di anomaly detection su dati reali di traffico cittadino, basato sui flussi orari rilevati dalle spire induttive del Comune di Bologna (open data dal 2019 al 2025), integrati con dati meteo storici da Open-Meteo e calendario delle festività italiane. Lo studente riceve una richiesta volutamente vaga da un committente fittizio: "vogliamo sapere quando succede qualcosa di insolito nel traffico", e deve trasformarla in un sistema funzionante: definire operativamente cosa considerare anomalia, distinguere tra anomalie del fenomeno e malfunzionamenti dei sensori sfruttando il dataset di accuratezza, costruire e confrontare tre approcci di natura diversa (decomposizione stagionale, metodi distance-based, forecasting), e infine progettare un protocollo di valutazione senza ground truth combinando iniezione di anomalie sintetiche, eventi cittadini storicamente noti e analisi di stabilità. Il progetto mette gli studenti alla prova soprattutto sulla formulazione del problema e sulla costruzione di metriche custom giustificate, più che sulla pura accuratezza modellistica.

## Dettagli

### 1. Lo scenario

Immaginate di essere stati contattati dal Comune di Bologna. L'amministrazione dispone di una rete di spire induttive installate sotto l'asfalto in tutta la città, che misurano ogni ora quanti veicoli passano in ogni punto monitorato. I dati sono pubblici e disponibili dal 2019 in poi. La richiesta che vi arriva è volutamente vaga:

"Vorremmo capire quando succede qualcosa di insolito nel traffico cittadino. Non sappiamo esattamente cosa cercare, ma sappiamo che ci sono giorni e ore in cui i flussi si comportano in modo che non ci aspettiamo, e vorremmo un sistema che ce li segnali."

Il vostro lavoro è trasformare questa richiesta in un sistema funzionante di rilevamento di anomalie, partendo dai dati grezzi. Il punto difficile non è il modello: è capire cosa state cercando, come misurate se l'avete trovato, e come lo comunicate a chi vi ha commissionato il lavoro.

Non vi viene fornito un dataset etichettato. Non c'è un "y vero" contro cui calcolare l'accuracy. Dovete costruirvi voi una nozione operativa di anomalia, giustificarla, e inventarvi un modo per validare il vostro sistema. Questo è esattamente lo scenario in cui vi troverete in un contesto professionale reale, ed è il vero oggetto di questo progetto.

### 2. Perché questo progetto non è quello che vi aspettate

Quasi tutti i tutorial di machine learning che avete visto seguono lo schema: dato un dataset etichettato, addestra un modello, calcola accuracy/precision/recall sul test set, riporta i risultati. Qui non funziona così, per tre ragioni.

Primo, non ci sono etichette. Nessuno ha annotato il dataset segnalando "qui c'è un'anomalia". Dovrete decidere voi cosa considerare anomalia, e questa decisione va difesa.

Secondo, esistono almeno due tipi di anomalia molto diversi nei dati, e confonderle è il modo più rapido per produrre un sistema inutile. C'è l'anomalia del fenomeno — il traffico è davvero anomalo, magari per un incidente, una manifestazione, un evento sportivo. E c'è l'anomalia dello strumento — il sensore si è guastato, è in manutenzione, sta riportando zero quando in realtà passavano macchine. Il committente vuole sapere della prima, non della seconda. Distinguerle è parte del lavoro.

Terzo, anche se il vostro modello funziona perfettamente sul piano statistico, dovete saper rispondere alla domanda: "come fate a sapere che funziona?". In assenza di etichette, le risposte plausibili sono: iniezione di anomalie sintetiche di cui conoscete la natura, validazione tramite eventi storici noti recuperati da fonti esterne, analisi di stabilità degli alert al variare degli iperparametri. Una di queste strategie va costruita esplicitamente.

### 3. I dati

#### 3.1 Dataset principale: flussi orari delle spire del Comune di Bologna

Il Comune di Bologna pubblica sul proprio portale Open Data le rilevazioni orarie del flusso veicolare misurate dalle spire induttive installate sul territorio. I dataset sono disponibili anno per anno dal 2019 al 2025 e seguono tutti lo stesso schema. Gli URL dei dataset annuali hanno la forma:
https://opendata.comune.bologna.it/explore/dataset/rilevazione-flusso-veicoli-tramite-spire-anno-AAAA/ dove AAAA è l'anno. 

Per ciascun anno è possibile esportare i dati in vari formati (CSV, JSON, GeoJSON) direttamente dal portale, oppure interrogare l'API Opendatasoft che il portale espone.

Struttura del record. Ogni riga rappresenta una spira in un giorno specifico. Le 24 fasce orarie sono in colonne separate (00:00-01:00, 01:00-02:00, ..., 23:00-24:00), ciascuna contenente il conteggio dei veicoli transitati in quell'ora. I campi di metadato includono la data, l'identificativo univoco della stazione spira (ID_univoco_stazione_spira — usate questo, e non codice_spira che può cambiare nel tempo per la stessa spira fisica), la via, la direzione di percorrenza, il nodo stradale di partenza e di arrivo, e le coordinate geografiche (longitudine e latitudine).
Operazione preliminare necessaria. Il formato wide con 24 colonne orarie non è quello che vi serve per fare time series analysis. Dovrete fare un melt/unpivot per portare i dati in formato long, con una colonna timestamp e una colonna conteggio_veicoli. Questo è il primo passaggio del vostro preprocessing.
 

#### 3.2 Dataset complementare: accuratezza delle spire

Per ogni anno di flussi esiste un dataset gemello che riporta l'accuratezza della misura. Gli URL hanno la forma:
https://opendata.comune.bologna.it/explore/dataset/accuratezza-spire-anno-AAAA/

Il valore di accuratezza è una percentuale tra 0 e 100, riferita alla stessa coppia (spira, ora). 100% significa che nella fascia oraria di riferimento la spira ha rilevato correttamente il dato per tutti i 60 minuti; 0% significa che non ha rilevato nulla per tutta la fascia; valori intermedi indicano rilevazione parziale.

Questo dataset non è un dettaglio tecnico, è il cuore del progetto. È il dato che vi permette di distinguere quando il sensore è guasto da quando il traffico è davvero anomalo. Una spira che riporta zero veicoli alle 17 di un mercoledì può essere un sensore rotto (accuratezza 0%) oppure una strada chiusa per emergenza (accuratezza 100% e zero veicoli effettivi). Sono due fenomeni diversissimi e il vostro sistema deve gestirli diversamente.
 

#### 3.3 Dato meteo
Open-Meteo offre un'API storica gratuita, senza chiave né registrazione, per scaricare dati meteo orari di qualsiasi località dal 1940 in poi. L'endpoint è: https://archive-api.open-meteo.com/v1/archive

Per Bologna le coordinate da usare sono approssimativamente latitudine 44.49, longitudine 11.34. Le variabili più rilevanti per il traffico sono temperature_2m, precipitation, rain, snowfall, wind_speed_10m, e potenzialmente weather_code per una classificazione qualitativa. La documentazione dell'endpoint è su https://open-meteo.com/en/docs/historical-weather-api.

Una singola chiamata HTTP GET con i parametri appropriati restituisce un JSON con tutte le ore richieste. 
 

#### 3.4 Calendario festività italiane

Per riconoscere automaticamente festività nazionali, weekend e giorni speciali, usate il pacchetto Python holidays, che si installa con pip install holidays e supporta l'Italia con la chiamata holidays.country_holidays('IT'). Per le festività locali bolognesi (es. San Petronio, 4 ottobre) e per eventi cittadini ricorrenti (manifestazioni, partite, fiere) dovrete costruirvi un piccolo calendario manuale: questa è parte del lavoro di domain knowledge.
 

#### 3.5 Eventi noti per la validazione

Per la fase finale di validazione vi servirà una lista di eventi cittadini realmente accaduti, da usare come ground truth proxy. Fonti consigliate: archivio del Resto del Carlino o di Repubblica Bologna per incidenti e blocchi del traffico, calendario delle partite del Bologna FC al Dall'Ara (impatta il traffico nella zona stadio), calendario delle fiere a BolognaFiere, comunicati del Comune su lavori stradali rilevanti. Non vi serve essere esaustivi: 15-20 eventi ben documentati su una finestra di test di alcuni mesi sono più che sufficienti.

### 4. Vincoli pratici e suggerimenti per non perdervi

Prima di entrare nelle fasi del progetto, alcune indicazioni operative che vi risparmiano giorni di frustrazione.

Non lavorate sull'intera rete di spire. Bologna ha decine di spire e l'intero dataset annuale può essere pesante. Selezionate un sottoinsieme di 15-25 spire scelte ragionando: alcune in centro, alcune sui viali, alcune in periferia, alcune sulla tangenziale, alcune su strade radiali in ingresso. 

Documentate la scelta nel report. Lavorare su tutte le spire non vi rende un progetto migliore, vi rende solo un progetto più lento.
Lavorate su un periodo significativo ma gestibile. Un buon punto di partenza è gennaio 2024 — settembre 2025: copre due regimi annuali completi, include estate (Ferragosto è uno stress-test importante), Natale, eventi cittadini noti, e vi lascia un margine di mesi recenti per il test set.

Evitate di mescolare 2020-2021 (Covid e lockdown) con anni normali se non sapete come gestire il cambio di regime.

Salvate i dati grezzi una volta sola. Scaricate tutti i CSV una volta, salvateli localmente, e da quel momento lavorate offline. Non rifate il download ad ogni esecuzione del notebook.

Fissate i seed random. In tutti gli script. Sempre. Senza eccezioni. La riproducibilità è criterio di valutazione.

### 5. Le fasi del progetto

Le quattro fasi corrispondono a circa una settimana ciascuna, ma sono flessibili. Quello che non è flessibile è l'ordine: non si parte dalla scelta del modello prima di aver definito il problema.
 

#### Fase 1 — Problem framing (settimana 1)

Prima di scrivere una singola riga di codice di modellazione, producete un documento di 2-4 pagine — il problem framing document — in cui rispondete esplicitamente alle seguenti domande.

Che cosa, per voi, in questo contesto, è una anomalia di traffico? Date una definizione operativa, non una generica. Vi state focalizzando su anomalie di tipo puntuale (una singola ora che si discosta dal pattern), contestuale (un'ora che dato il giorno della settimana e il meteo si discosta), o collettiva (una sequenza di ore che insieme formano un pattern strano)? Quale combinazione di queste?

Quali tipi di anomalia il sistema deve catturare e quali esplicitamente non deve catturare? Per esempio: il sistema deve flaggare i guasti dei sensori? Probabilmente no, perché il committente vuole sapere del traffico, ma deve almeno saperli riconoscere e gestire diversamente. Il sistema deve flaggare il calo strutturale di traffico ad agosto per le ferie? Probabilmente no, perché è atteso e noto.

Chi è l'utente finale immaginato del sistema? Un dispatcher di polizia municipale che riceve un alert in tempo reale? Un analista che fa report settimanali? Un sistema automatico che attiva semafori adattivi? Le tre risposte implicano tre sistemi molto diversi in termini di latenza, sensibilità, granularità.

Qual è il costo di un falso positivo rispetto a un falso negativo? Se il sistema invia un alert per niente, cosa succede? Se manca un evento reale, cosa succede? Questa asimmetria guiderà la scelta delle soglie più avanti.

Questo documento è vincolante per tutto il resto del progetto. Le scelte di Fase 2 e Fase 3 dovranno essere coerenti con quanto deciso qui. Se cambiate idea strada facendo, va benissimo, ma il cambiamento deve essere esplicito e motivato nel report finale.
 

#### Fase 2 — Esplorazione e baseline (settimana 2)

##### 2.a EDA orientata al problema

L'analisi esplorativa non è un esercizio di stile. Ogni grafico che producete deve rispondere a una domanda. Suggerimenti di domande utili.
Come si comporta il traffico nelle 24 ore di un giorno medio, separando per giorno della settimana? Quanto è regolare il pattern? Quanto varia tra spire diverse della stessa categoria (es. due spire entrambe in centro)?

Esiste una stagionalità settimanale (lun-ven vs sab-dom)? Una stagionalità annuale (estate vs inverno, agosto)? Le due stagionalità interagiscono?
Quanto e come il meteo influenza il traffico? La pioggia aumenta o diminuisce i flussi? La risposta non è ovvia e dipende dalla strada.
Quanto sono "rumorosi" i dati? Calcolate la deviazione standard ora-per-ora rispetto al pattern stagionale: dove vivono le anomalie naturalmente non rilevate?

Qual è la distribuzione del dataset di accuratezza? Quante ore-spira hanno accuratezza inferiore a 100%? A 50%? A 0%? Queste vanno trattate, e dovete decidere come (escluderle? imputarle? marcarle?).

##### 2.b Costruzione di tre baseline

Costruite tre approcci di natura diversa all'anomaly detection, in ordine di complessità crescente.

- Baseline 1 — Statistica classica. Decomposizione della serie storica in trend, stagionalità multipla (giornaliera + settimanale) e residuo, usando ad esempio statsmodels.tsa.seasonal_decompose o STL (statsmodels.tsa.STL). Soglia sui residui basata su scarti dalla mediana, usando il MAD (mediana degli scarti assoluti) che è più robusto della deviazione standard. Punto anomalia se il residuo standardizzato supera una soglia (es. 3 o 4).

- Baseline 2 — Distance/density-based. Costruzione di feature engineered (ora del giorno, giorno della settimana, lag di 1h/24h/168h, meteo) e applicazione di IsolationForest o LocalOutlierFactor da sklearn.ensemble / sklearn.neighbors. Calibrazione della contamination rate motivata dal framing iniziale.

- Baseline 3 — Forecasting-based. Un modello che predice il valore atteso di una spira per una specifica ora dato il contesto (lag, giorno, meteo) — un regressore tipo GradientBoostingRegressor, HistGradientBoostingRegressor o LightGBM va benissimo. Lo scarto tra valore osservato e valore predetto, normalizzato per l'errore tipico del modello, è il vostro punteggio di anomalia.

Per ciascuna baseline rispondete nel report a queste tre domande: che tipi di anomalia cattura naturalmente questo approccio? Che tipi non cattura? Su quali ipotesi sui dati si basa, e queste ipotesi sono verificate?
 

#### Fase 3 — Sistema integrato e protocollo di valutazione (settimana 3)

##### 3.a Composizione del sistema finale

Mettete insieme un sistema che combini in modo motivato i tre approcci. Le opzioni ragionevoli sono diverse e tutte accettabili: un ensemble che fa media o massimo dei punteggi, una cascata in cui il primo modello filtra grossolanamente e il successivo raffina, un meta-modello che pesa diversamente i tre punteggi a seconda del contesto, o anche una scelta motivata di uno solo dei tre se argomentate perché gli altri non aggiungono valore.
Non importa quanto sia sofisticato. Importa che la scelta sia motivata in modo coerente con il framing della Fase 1.

##### 3.b Costruzione del protocollo di valutazione

Questa è la parte più importante e meno guidata del progetto. Vi suggerisco di combinare almeno due delle seguenti strategie.

Iniezione di anomalie sintetiche. Costruite una tassonomia di anomalie plausibili (es. picco improvviso di flusso, drop improvviso, plateau anomalo, valori shiftati di una costante) e iniettatele in posizioni note del test set. Misurate poi la capacità del sistema di recuperarle. Questa è la strategia più pulita ma soffre del problema che le anomalie sintetiche potrebbero essere troppo facili o troppo diverse da quelle vere.

Validazione con eventi reali. Recuperate 15-20 eventi storici noti e verificate se il sistema ha prodotto alert in corrispondenza, su quali spire, con che intensità. Questa è la strategia più realistica ma soffre di copertura limitata.

Analisi di stabilità. Fate variare gli iperparametri principali (soglie, contamination rate, finestre temporali) e osservate come cambia l'insieme degli alert prodotti. Un sistema robusto produce un nucleo stabile di alert con piccole variazioni ai margini; un sistema fragile produce insiemi completamente diversi. Plot di sensibilità sono molto utili nel report.

Metriche custom motivate. Costruite almeno una metrica composita che rifletta il framing iniziale, e non limitatevi a precision/recall su soglia fissa. Una metrica AUC sulla curva di precision-recall calcolata contro la ground truth proxy è un punto di partenza decente. Se nel framing avete dichiarato che i falsi positivi costano molto, una metrica F-beta con beta < 1 è coerente.

Quello che non è accettabile come unica strategia di valutazione: riportare la contamination rate del modello come se fosse la performance, dichiarare "il modello ha identificato il 5% di anomalie" senza dire se quel 5% è giusto, calcolare precision/recall contro le anomalie sintetiche e basta, o ignorare il problema dichiarando che "in assenza di etichette la valutazione non è possibile".
 

#### Fase 4 — Reporting e consegna (settimana 4)

L'ultima settimana è dedicata a scrivere il report e ripulire il codice. Non sottovalutate questo tempo. Un progetto eccellente ma comunicato male perde metà del proprio valore.

## 6. Cosa consegnare
La consegna consiste di due elementi obbligatori.
 

### 6.1 Repository di codice riproducibile

Repository Git (GitHub, GitLab, o cartella zippata) contenente tutto il codice del progetto. Struttura suggerita:

project_work_traffico/
├── README.md              # come riprodurre tutto, da zero
├── requirements.txt       # versioni esatte dei pacchetti
├── data/
│   ├── raw/              # dati grezzi scaricati (o script per scaricarli)
│   └── processed/        # dati lavorati
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_sistema_finale.ipynb
│   └── 04_valutazione.ipynb
├── src/                  # funzioni riutilizzabili
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── results/              # output: figure, tabelle, csv di alert
└── docs/
    └── problem_framing.md  # documento della Fase 1

I notebook Jupyter sono ammessi e anzi consigliati per esplorazione e visualizzazione. Tutti i seed devono essere fissati. Il README deve permettere a chi non sa nulla del progetto di clonare il repository e rieseguire tutto.
 

### 6.2 Report tecnico

Documento in PDF o Microsoft Word. Sezioni richieste:
- Introduzione e contesto. Descrizione dello scenario, della domanda di business, dell'approccio generale che avete scelto. Non riassumete la traccia: presentate la vostra interpretazione di essa.
- Problem framing. Trasposizione del documento della Fase 1, eventualmente raffinato alla luce di quanto scoperto strada facendo. Tutte le scelte successive devono trovare radice qui.
- Dati e preprocessing. Descrizione delle fonti, della selezione di spire, della finestra temporale, del trattamento dei dati di accuratezza, delle integrazioni con meteo e festività. Statistiche descrittive essenziali.
- Esplorazione. Le evidenze chiave emerse dall'EDA, con i grafici che le supportano. Non tutta l'EDA che avete fatto, solo quella rilevante per le scelte modellistiche.
- Modelli. Le tre baseline e il sistema finale, con motivazione delle scelte. Niente walkthrough del codice, ma scelte di design e iperparametri.
Valutazione. Il protocollo di valutazione, le metriche, i risultati. Tabelle e plot dei risultati. Confronto tra approcci.

Limiti e discussione critica (sezione obbligatoria e valutata pesantemente). Quali sono i limiti del vostro sistema? In quali condizioni fallirebbe? Quali assunzioni sui dati avete fatto che potrebbero non reggere su dati nuovi? Quali categorie di anomalie non riuscite a catturare? In quali scenari operativi il sistema non dovrebbe essere usato?

Conclusioni e lavori futuri. Cosa portereste a un secondo round di sviluppo, con priorità.
Bibliografia. Citate per esteso fonti, paper, tutorial, documentazione consultati.
Appendici (facoltative). Tabelle di dettaglio, ulteriori plot, log di esperimenti.

## 7. Criteri di valutazione

La griglia di valutazione, espressa in pesi percentuali, premia in modo deliberato gli aspetti che rendono il progetto formativo e non solo tecnico.
Problem framing e coerenza interna (30%). Quanto è chiaro, ragionato e operativo il framing della Fase 1, e quanto le scelte successive sono coerenti con esso. Un framing vago o ignorato dalle fasi successive è il problema più grave possibile in questo progetto.

Protocollo di valutazione e analisi critica (30%). Quanto è solido il modo in cui valutate il vostro sistema in assenza di etichette. La qualità della sezione "Limiti e discussione critica" rientra qui.

Solidità tecnica (25%). Qualità del preprocessing, sensatezza delle scelte modellistiche, correttezza dell'implementazione, riproducibilità del codice, gestione corretta del dataset di accuratezza, integrazione tra fonti dati.

Qualità della comunicazione (15%). Chiarezza del report, qualità dei grafici, leggibilità del codice, qualità del README. Un buon report di un progetto modesto vale più di un report confuso di un progetto eccellente.

L'accuratezza assoluta del modello non è criterio di valutazione. Un sistema che produce risultati mediocri ma ben argomentato e onesto sui propri limiti vale più di un sistema che sembra eccellente ma non sa rispondere alla domanda "come fai a saperlo?".

## 8. Trappole comuni che vi auguriamo di incontrare

Le elenco perché vi aspetto a una di queste, cascarci non è un problema, è esperienza. Ignorarle nel report sì.

Trattare come anomalia qualunque deviazione dalla media, dimenticandosi che il traffico ha stagionalità multipla e quello che sembra anomalo a colpo d'occhio è in realtà perfettamente normale per un sabato sera o per Ferragosto.

Confondere anomalia del fenomeno con anomalia del sensore. Il vostro sistema flaggerà inevitabilmente molte ore-spira con accuratezza bassa: questo non è successo, è failure mode previsto. La domanda è cosa farete per gestirlo.

Costruirsi un test set troppo facile iniettando anomalie sintetiche enormi e ovvie, ottenere recall del 100%, e dichiararsi soddisfatti. Le anomalie sintetiche utili sono quelle al confine della rilevabilità.

Usare soglie statiche sulla deviazione standard di una serie altamente eteroschedastica (cioè in cui la varianza dipende dall'ora del giorno). Le 3 del mattino e le 18 hanno scale di rumore diversissime; una soglia unica è sbagliata in entrambe.

Dimenticarsi del fuso orario nel join tra dati di traffico (timezone Europa/Roma) e dati Open-Meteo (default UTC). Un offset di 1-2 ore può rendere completamente inutile la feature meteo.

Usare future information come feature, per esempio facendo train/test split casuale anziché temporale, o normalizzando il dataset intero prima dello split. Il vostro test set deve essere strettamente posteriore al training set in tempo.

## 9. Risorse di partenza

Documentazione dataset Bologna:
Pagina sintetica: https://dati.comune.bologna.it/dati/flusso-veicolare
Flussi anno 2024: https://opendata.comune.bologna.it/explore/dataset/rilevazione-flusso-veicoli-tramite-spire-anno-2024/
Accuratezza anno 2024: https://opendata.comune.bologna.it/explore/dataset/accuratezza-spire-anno-2024/

API meteo:
Documentazione Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
Pacchetti Python di base che userete: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, holidays, requests, e a vostra scelta lightgbm o xgboost per la baseline 3.

Buon lavoro. Ricordatevi che il committente esiste anche se è immaginario: scrivete il report come se dovesse leggerlo lui, non come se dovesse leggerlo solo il docente.