# Predire l'imprevedibile

Project work di **anomaly detection** sui flussi orari delle spire induttive del Comune di Bologna (open data 2024-2025), integrati con dati meteo (Open-Meteo) e calendario festività italiane.

> *"Vorremmo capire quando succede qualcosa di insolito nel traffico cittadino."*
> — Richiesta volutamente vaga del committente

In assenza di etichette di ground truth, il lavoro consiste nel trasformare la richiesta in un sistema funzionante: definire operativamente cosa è anomalia, costruire più approcci di rilevamento, e progettare un protocollo di valutazione.

## Indice rapido

- **[Piano di lavoro](PIANO.md)** — la roadmap progettuale (Fase 0)
- **[Problem framing](problem_framing.md)** — Fase 1: definizione operativa dell'anomalia
- **Notebooks** — i 4 notebook tecnici, dall'EDA alla valutazione finale
    - [01 - Esplorazione dei dati](notebooks/01_eda.ipynb)
    - [02 - Tre baseline](notebooks/02_baselines.ipynb)
    - [03 - Sistema ensemble](notebooks/03_sistema_finale.ipynb)
    - [04 - Protocollo di valutazione](notebooks/04_valutazione.ipynb)
- **[Report tecnico](report_tecnico.md)** — Fase 4: sintesi, risultati e discussione critica

## L'approccio in 4 fasi

| Fase | Output | Riferimento |
|---|---|---|
| **1. Problem framing** | Definizione operativa, tipi target, asimmetria costi FP/FN | [problem_framing.md](problem_framing.md) |
| **2.a EDA** | Stagionalità, eteroschedasticità, qualità dato | [01_eda.ipynb](notebooks/01_eda.ipynb) |
| **2.b Tre baseline** | STL+MAD, IsolationForest, LightGBM | [02_baselines.ipynb](notebooks/02_baselines.ipynb) |
| **3.a Sistema** | Ensemble per max dei rank normalizzati | [03_sistema_finale.ipynb](notebooks/03_sistema_finale.ipynb) |
| **3.b Valutazione** | Sintetiche + eventi reali + stabilità | [04_valutazione.ipynb](notebooks/04_valutazione.ipynb) |
| **4. Reporting** | Documento tecnico, limiti, lavori futuri | [report_tecnico.md](report_tecnico.md) |

## Risultati di sintesi

- **Hit rate eventi reali**: 77.8% (14 su 18 eventi storici colpiti)
- **Recall anomalie sintetiche**: 64.5% sui drop, 62.1% sugli spike
- **Stabilità**: J(P99, P99.5) = 0.50 → nucleo di alert robusto
- **Scoperta documentata**: B3 (LightGBM) da solo supera l'ensemble, perché B2 (IsolationForest) introduce rumore. Trasparenza > ottimizzazione cosmetica.

## Come riprodurre

```bash
git clone https://github.com/fedcal/Rilevamento-anomalie-nel-traffico-di-Bologna.git
cd Rilevamento-anomalie-nel-traffico-di-Bologna
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python -m src.data_loading    # scarica ~40MB di open data
./venv/bin/python -m src.preprocessing   # costruisce il master dataset
# Esegui i notebook nell'ordine
./venv/bin/jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
```

Vedi il [README del repository](https://github.com/fedcal/Rilevamento-anomalie-nel-traffico-di-Bologna#readme) per istruzioni complete.

## Stack tecnico

`pandas` · `numpy` · `scikit-learn` · `statsmodels` (STL) · `lightgbm` · `matplotlib` · `seaborn` · `holidays` · Jupyter
