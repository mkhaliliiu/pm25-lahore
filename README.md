# Forecasting PM2.5 and Predicting Severe Smog Events in Lahore

Machine-learning forecasts of hourly PM2.5 **24, 48 and 72 hours ahead** and early warnings of
**severe smog** (24-hour mean above 125.5 µg/m³, US EPA "Very Unhealthy") for Lahore, Pakistan.
The models correct the CAMS global air-quality forecast with local measurements and weather, and every
result comes from an out-of-sample **rolling-origin backtest**.

*Data Science Tools and Techniques (DSTT) semester project, FAST NUCES Islamabad, Fall 2026.*
*Muhammad Khalil (26i-8033) and Muhammad Shakil (26i-8024). Instructor: Dr Mateen Yaqoob.*

## Main results (10 folds, Aug 2023 to Mar 2025)

| | 24 h | 48 h | 72 h |
|---|---|---|---|
| Persistence, MAE (µg/m³) | 62.1 | 70.8 | 74.4 |
| Raw CAMS, MAE | 87.7 | 87.9 | 88.3 |
| **Ridge + CAMS, MAE** | **53.8** | **56.8** | **57.8** |
| LSTM, MAE | 55.4 | 57.5 | 58.8 |

- **RQ1:** The tuned LightGBM and the LSTM beat persistence, climatology and raw CAMS at every horizon
  (Diebold-Mariano p ≤ 0.002), and Ridge and Random Forest have lower error still.
  CAMS is strongly biased low in Lahore (mean bias −63 µg/m³, −114 in the smog season). Adding it as a
  feature lowers LightGBM error significantly at every horizon (p ≤ 0.03). Simple models (Ridge, Random Forest)
  matched or beat boosted trees and the LSTM.
- **RQ2:** A random 80/20 split **understates the error by 31–35%** compared with the rolling-origin backtest.
- **RQ3:** With a 10:1 cost for a missed event, LightGBM + SMOTE at 48 h reaches recall 0.97 and precision 0.87.
  Its cost is 2,572, against 6,844 for persistence. It flagged all 7 smog episodes that began after three
  clean days at least 48 h in advance; persistence and raw CAMS flagged none.
  Choosing the threshold by cost mattered more than resampling. SMOTE helped most for the rarer
  "Hazardous" (> 225.5) event.

## Live forecast

The US Embassy monitor's public feed ended in March 2025, so measured PM2.5 is no longer available in
real time. The dashboard's **Live forecast** tab and the API's `/live` endpoint therefore use a separate
**live model**. It learns a correction of the CAMS forecast for Lahore from inputs that
[Open-Meteo](https://open-meteo.com) provides live and without a key: current weather, the current CAMS
value, and the CAMS forecast for the target hour. It is backtested on the same folds and test hours
(`notebooks/07_live_model.py`, `results/live_model_*.csv`), so its accuracy is reported honestly next
to raw CAMS and the full model.

Full tables are in [`results/`](results) and figures in [`results/figures/`](results/figures).

## Repository layout

```
notebooks/   00_colab_setup  01_data_acquisition  02_preprocess_eda  03_regression
             04_classification  05_lstm_shap  06_export_models  07_live_model   (run in this order)
src/         config.py (paths, secrets)  features.py  models.py  evaluation.py  deploy.py  live.py
app/         streamlit_app.py, requirements.txt, data/ (small out-of-sample snapshots)
api/         main.py (FastAPI)
models/      exported models: Ridge as JSON, LightGBM as text, meta.json (full) and
             meta_live.json (live). No pickle files.
results/     metrics (CSV/JSON) and figures produced by the notebooks
```

## Run the dashboard and API (Python 3.10+)

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py          # http://localhost:8501
uvicorn api.main:app --reload               # http://127.0.0.1:8000/docs
```

API endpoints: `GET /live` (today's forecast), `GET /health`, `GET /forecast?time=2025-03-01T08:00`, `GET /features/example`,
`POST /predict` with `{"issued_at": "...", "features": {...}}`.

With Docker:

```bash
docker build -t pm25-lahore .
docker run -p 8501:8501 pm25-lahore
docker run -p 8000:8000 pm25-lahore uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Reproduce the results (Google Colab)

1. Get a free OpenAQ API key and store it as the Colab secret `OPENAQ_API_KEY`. For local runs, copy
   `.env.example` to `.env` and put it there. `.env` is git-ignored and keys are never printed.
2. Run `notebooks/00_colab_setup.py` (mounts Drive, clones the repo, installs `requirements.txt`).
3. Run the notebooks in order with `%run notebooks/<name>.py`. Approximate times on Colab:
   03 about 22 min, 04 about 18 min, 05 about 3 min on a T4 GPU,
   06 about 3 min, 07 about 10 min.

Large intermediate files (`data/`) are not committed; notebook 01 re-creates them.

## Data sources

- **PM2.5:** US Diplomatic Post Lahore via [OpenAQ](https://openaq.org) (sensor 25135), hourly,
  May 2019 to Mar 2025. The station's public feed ends in March 2025, so the deployed demo runs on
  stored data.
- **Weather:** ERA5 reanalysis via the [Open-Meteo](https://open-meteo.com) historical API.
- **CAMS:** Copernicus Atmosphere Monitoring Service global PM2.5 forecast via the Open-Meteo
  Air Quality API (available from Aug 2022).

## Limitations

- The data comes from a single monitor.
- Archived CAMS values may not match the lead time of a real-time CAMS forecast.
- Tree-model probabilities are poorly calibrated, so the cost-optimal thresholds are very low
  and give more false alarms than persistence.
- Only 7 to 16 smog-episode onsets fall in the test period, too few for firm lead-time conclusions.
