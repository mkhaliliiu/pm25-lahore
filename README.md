# Forecasting PM2.5 and Predicting Severe Smog Events in Lahore Using Machine Learning on Imbalanced Data

This project forecasts hourly PM2.5 in Lahore **24, 48 and 72 hours ahead** and gives early warnings of
**severe smog**, defined as a 24-hour mean PM2.5 above 125.5 µg/m³ (the US EPA "Very Unhealthy" level).
The models correct the global CAMS air-quality forecast with local measurements and weather data.
All reported results come from an out-of-sample **rolling-origin backtest** (10 folds, August 2023 to March 2025).
The final models are deployed as a live Streamlit dashboard and a FastAPI service.

| | |
|---|---|
| **Course** | Data Science Tools and Techniques (DSTT), MS Data Science, FAST NUCES Islamabad, Fall 2026 |
| **Instructor** | Dr Mateen Yaqoob |
| **Team** | Muhammad Khalil (26i-8033), Muhammad Shakil (26i-8024) |
| **Live dashboard** | https://pm25-lahore.streamlit.app/ |
| **Full run log (Colab)** | [`Colab Code file/PM25_Main.ipynb`](Colab%20Code%20file/PM25_Main.ipynb), with the outputs of every step |
| **Results** | [`results/`](results) (tables as CSV/JSON) and [`results/figures/`](results/figures) (all plots) |

---

## Contents

1. [Main results](#1-main-results)
2. [Repository structure](#2-repository-structure)
3. [How the project works](#3-how-the-project-works)
4. [How to run the project](#4-how-to-run-the-project)
5. [Notebook guide (what each notebook does)](#5-notebook-guide-what-each-notebook-does)
6. [The Colab notebook `PM25_Main.ipynb`, cell by cell](#6-the-colab-notebook-pm25_mainipynb-cell-by-cell)
7. [Results folder guide](#7-results-folder-guide)
8. [Figures folder guide](#8-figures-folder-guide)
9. [Where each table and figure of the paper comes from](#9-where-each-table-and-figure-of-the-paper-comes-from)
10. [Models, app data and the data folder](#10-models-app-data-and-the-data-folder)
11. [Live forecasting system](#11-live-forecasting-system)
12. [Dashboard and API](#12-dashboard-and-api)
13. [Security of API keys](#13-security-of-api-keys)
14. [Data sources](#14-data-sources)
15. [Limitations](#15-limitations)

---

## 1. Main results

**PM2.5 forecast error (MAE in µg/m³; 10 rolling-origin folds, Aug 2023 to Mar 2025)**

| Model | 24 h | 48 h | 72 h |
|---|---|---|---|
| Persistence (baseline) | 62.1 | 70.8 | 74.4 |
| Raw CAMS forecast (baseline) | 87.7 | 87.9 | 88.3 |
| **Ridge + CAMS (deployed model)** | **53.8** | **56.8** | **57.8** |
| Random Forest + CAMS | 53.6 | 57.6 | 58.5 |
| LightGBM + CAMS (Optuna-tuned) | 55.8 | 60.7 | 61.8 |
| LSTM | 55.4 | 57.5 | 58.8 |

Full table: `results/regression_summary.csv`; the LSTM rows are in `results/lstm_comparison.csv`.

- **RQ1 (ML vs CAMS):** every model with CAMS inputs beat all four baselines (persistence, weekly naive,
  climatology, raw CAMS). The tuned LightGBM was 26 to 32 µg/m³ more accurate than raw CAMS
  (Diebold-Mariano p ≤ 0.002 against all baselines). Ridge + CAMS had the lowest error at 48 and 72 h:
  34 to 39% below raw CAMS and 14 to 22% below persistence. Raw CAMS is strongly biased low in Lahore
  (mean bias −63 µg/m³, and −114 µg/m³ in the smog season).
- **RQ2 (random split vs rolling origin):** scored on the same test hours, a shuffled split understated the
  error of tuned LightGBM by 23 to 29%, but Ridge's by only 5 to 6%. The random split also reversed the
  model ranking. Over all hours since Aug 2022 (a simple 80/20 split), the understatement was 31 to 35%.
- **RQ3 (severe-smog warnings):** with a missed event costing 10 false alarms, LightGBM + SMOTE at 48 h
  reached recall 0.968 and precision 0.865. Its cost was 2,572, against 6,844 for persistence. It flagged
  all 7 episodes that began after three clean days at least 48 h in advance; persistence and raw CAMS
  flagged none. SMOTE had the lowest cost in 8 of 9 horizon/cost-ratio combinations; the exception is the
  plain model at 24 h and 10:1. The price of this recall is 2.2 to 3.7 false-alarm days per month,
  against 0.6 to 1.2 for persistence.
- **Cost-sensitive regression (instructor's suggestion):** we gave November and other high-peak hours
  larger training weights. This lowered the LightGBM error on hours ≥ 125.5 µg/m³ by only 2 to 5 µg/m³
  and raised the error on clean hours by 20 to 24%. The selection rule, fixed in advance, therefore
  kept the deployed models unweighted.
- **Second station:** filling gaps in the PM2.5 inputs with a calibrated Clarity sensor changed MAE by
  at most 0.3 µg/m³.
- **Live check (8 to 15 Nov 2026, 169 hours):** 24-h forecasts had MAE 60 µg/m³ against the calibrated
  sensor network, versus 84 µg/m³ for raw CAMS.

---

## 2. Repository structure

```
pm25-lahore/
├── Colab Code file/
│   └── PM25_Main.ipynb      Colab notebook that ran every step, with all outputs saved (run log)
├── notebooks/               The analysis, one script per step (00 to 12), run in order
├── src/                     Shared code imported by the notebooks, app and API
│   ├── config.py            Paths, constants (location, dates, thresholds), secret loading
│   ├── features.py          Cleaning, feature engineering, targets (PM2.5 at t+h, smog event)
│   ├── models.py            Model definitions (Ridge, RF, XGBoost, LightGBM, SMOTE + LightGBM)
│   ├── evaluation.py        Rolling-origin folds, metrics, Diebold-Mariano test, bootstrap, cost thresholds
│   ├── deploy.py            Loads exported models, US AQI conversion, prediction helpers
│   └── live.py              Live data: OpenAQ sensor network, Open-Meteo forecast, verification
├── results/                 All metrics tables (CSV/JSON) produced by the notebooks
│   └── figures/             All plots (PNG) produced by the notebooks
├── models/                  Exported final models (Ridge as JSON, LightGBM as text, no pickle files)
├── app/
│   ├── streamlit_app.py     Dashboard (6 tabs), deployed on Streamlit Cloud
│   ├── requirements.txt     Light requirements for the dashboard and API
│   └── data/                Small snapshots used by the dashboard (backtest, observations, sensor list)
├── api/main.py              FastAPI service (/live, /forecast, /predict, ...)
├── data/raw/                Downloaded data (git-ignored; re-created by notebook 01)
├── Dockerfile               Container for the dashboard and API
├── requirements.txt         Full requirements for running the notebooks
├── .env.example             Template for the local .env file (the real .env is git-ignored)
└── .gitignore
```

---

## 3. How the project works

```
 Data (01)  ──►  Cleaning + features + EDA (02)  ──►  Forecast models (03, 05)  ──►  Smog warnings (04)
                                                              │
                                                              ▼
       Extra checks (09, 11, 12)  ◄──  Evaluation: 10-fold rolling origin, DM tests, bootstrap
                                                              │
                                                              ▼
              Export final models (06)  ──►  Live model and sensor network (07, 08, 10)  ──►  Dashboard + API
```

1. **Data.** Hourly PM2.5 from the US Diplomatic Post monitor in Lahore (OpenAQ, May 2019 to Mar 2025),
   ERA5 weather and the CAMS PM2.5 forecast (both through Open-Meteo, CAMS available from Aug 2022).
   All times are Pakistan time (UTC+5).
2. **Cleaning and features.** We remove non-positive and stuck readings and fill gaps of up to 3 h
   (inputs only). We build 30 inputs: PM2.5 lags and rolling statistics, weather, ventilation, calendar,
   and the CAMS forecast for the target hour.
3. **Forecasting.** A separate model for each horizon (24, 48, 72 h): four baselines, Ridge,
   Random Forest, XGBoost, LightGBM (default and Optuna-tuned) and an LSTM.
4. **Warnings.** LightGBM classifiers (plain, class weights, SMOTE) predict whether the 24-h mean will exceed
   125.5 µg/m³. The decision threshold minimises cost (a missed event = 10 false alarms) and is chosen
   on training data only.
5. **Evaluation.** Ten 2-month test blocks from Aug 2023 to Mar 2025. Before each block the models are
   retrained on all earlier data (expanding window), so no test hour is ever seen in training.
6. **Deployment.** The chosen models (Ridge + CAMS for PM2.5, LightGBM + SMOTE for warnings) are exported
   and served by the dashboard and API. The live forecast uses a calibrated network of low-cost sensors,
   because the US Embassy monitor's public feed ended in March 2025.

---

## 4. How to run the project

### Option A: just look at it (no setup)

- Open the live dashboard: https://pm25-lahore.streamlit.app/
- Open [`Colab Code file/PM25_Main.ipynb`](Colab%20Code%20file/PM25_Main.ipynb) on GitHub. It shows the
  saved output of every step, exactly as it ran.
- Browse [`results/`](results) and [`results/figures/`](results/figures).

### Option B: reproduce everything in Google Colab (recommended)

The heavy steps were run in Google Colab because they take 10 to 25 minutes each.

**Step 1: Colab secrets.** In Colab, open the key icon (Secrets) in the left sidebar and add:

| Secret | Needed for | Where to get it |
|---|---|---|
| `OPENAQ_API_KEY` | notebooks 01, 08, 09 (part E), 10 and the sensor-check cells | free account at https://explore.openaq.org (Account, API key) |
| `GITHUB_TOKEN` | only for cloning a private repo and pushing results (our own workflow) | GitHub, fine-grained token, this repo only |

Turn on "Notebook access" for each secret.

**Step 2: open the notebook and run it.** Open `Colab Code file/PM25_Main.ipynb` in Colab
(File, Upload notebook, or File, Open notebook, GitHub tab). In Cell 1, edit the settings to your own
GitHub user and Drive folder. Then run the cells from top to bottom. Cell 2 mounts Google Drive, clones
the repo into Drive and moves into it, so data and results survive a Colab disconnect.
Section 6 explains every cell.

**Shorter version (no push rights needed).** In a new Colab notebook:

```python
# Clone (or upload and unzip the repository zip, then %cd into it)
!git clone https://github.com/mkhaliliiu/pm25-lahore.git
%cd pm25-lahore
%pip install -q -r requirements.txt

# The OpenAQ key is read automatically from the Colab secret OPENAQ_API_KEY
%run notebooks/01_data_acquisition.py     # download data            (needs OpenAQ key)
%run notebooks/02_preprocess_eda.py       # clean, features, EDA
%run notebooks/03_regression.py           # PM2.5 models, RQ1 + RQ2   (~15-25 min)
%run notebooks/04_classification.py       # smog warnings, RQ3        (~15-25 min)
%run notebooks/05_lstm_shap.py            # LSTM + SHAP               (use a T4 GPU runtime)
%run notebooks/06_export_models.py        # final models + app data
%run notebooks/07_live_model.py           # weather + CAMS live model
# sensor check: run cells 19-20 of PM25_Main.ipynb (writes data/raw/openaq_lahore_active_sensors.csv)
%run notebooks/08_live_network.py         # live sensors, hourly models (needs OpenAQ key)
%run notebooks/09_proposal_checks.py      # remaining proposal items
%run notebooks/10_clarity_bridge.py       # bridge calibration, second station (needs OpenAQ key)
%run notebooks/11_rq2_same_period.py      # RQ2 on the same test hours
%run notebooks/12_cost_sensitive.py       # peak-weighted (cost-sensitive) regression
```

Notebooks must be run from the repository root, because each one imports `src/`. If the repository
is private, use a token in the clone URL, or upload the repository zip to Colab and unzip it.

### Option C: run locally (Python 3.10 or newer)

**Dashboard and API only** (uses the committed models and data, no training needed):

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py      # dashboard at http://localhost:8501
uvicorn api.main:app --reload           # API at http://127.0.0.1:8000/docs
```

For the live tab to read the sensor network, copy `.env.example` to `.env` and add your `OPENAQ_API_KEY`.
Without a key, the dashboard still works and uses the weather + CAMS fallback model.

**With Docker:**

```bash
docker build -t pm25-lahore .
docker run -p 8501:8501 pm25-lahore                                                   # dashboard
docker run -p 8000:8000 pm25-lahore uvicorn api.main:app --host 0.0.0.0 --port 8000   # API
```

**Full pipeline locally:**

```bash
pip install -r requirements.txt          # notebook 05 also needs TensorFlow (pip install tensorflow)
cp .env.example .env                     # then put your OpenAQ key in .env
python notebooks/01_data_acquisition.py
python notebooks/02_preprocess_eda.py
# ... and so on, in order, up to 12
```

**Note on re-running:** random seeds are fixed (42), so the modelling steps are repeatable. Notebooks 01, 08, 09
and 10 download data from OpenAQ and Open-Meteo, which can revise past values. The live sensor
network also changes over time, so a fresh run can give slightly different numbers or a different
sensor list than the committed results.

---

## 5. Notebook guide (what each notebook does)

The notebooks are Python scripts with `# %%` cell markers, so they run both as scripts and as Colab
cells (`%run notebooks/<name>.py`). Each file starts with a header describing its inputs and outputs.

| # | File | What it does | Needs | Time (Colab CPU) | Main outputs |
|---|---|---|---|---|---|
| 00 | `00_colab_setup.py` | Colab setup: mount Drive, clone the repo, install packages, safe git pull/push helpers | Colab secrets | short | none |
| 01 | `01_data_acquisition.py` | Downloads US Embassy PM2.5 (OpenAQ), ERA5 weather and CAMS forecast (Open-Meteo), and merges them on one hourly grid (Pakistan time) | OpenAQ key | a few minutes | `data/raw/*.csv` (git-ignored) |
| 02 | `02_preprocess_eda.py` | Cleaning (non-positive and stuck readings, short gaps), 30 features, targets, data summary and EDA plots | 01 | short | `data/features.csv`, `results/data_summary.json`, 5 EDA figures |
| 03 | `03_regression.py` | **RQ1 and RQ2.** Rolling-origin backtest of all PM2.5 models at 24/48/72 h, Optuna tuning, Diebold-Mariano tests, bootstrap intervals, random 80/20 split comparison | 02 | 15–25 min | `regression_*.csv`, `dm_tests.csv`, `bootstrap_ci.csv`, `rq2_random_vs_rolling.csv`, `optuna_best_params.json` |
| 04 | `04_classification.py` | **RQ3.** Severe-smog warnings: LightGBM plain vs class weights vs SMOTE, cost-based thresholds (5:1, 10:1, 20:1), Hazardous (> 225.5) check, lead time at episode onsets | 03 | 15–25 min | `classification_summary.csv`, `classification_by_season.csv`, `lead_time.csv` |
| 05 | `05_lstm_shap.py` | LSTM (Keras) on the same folds and test hours; SHAP explanation of LightGBM + CAMS | 03, GPU recommended | ~10 min on T4 GPU | `lstm_comparison.csv`, `lstm_dm_tests.csv`, `shap_importance.csv` |
| 06 | `06_export_models.py` | Trains the final models on all data and exports them without pickle (Ridge + CAMS, LightGBM + SMOTE); writes the dashboard snapshots and the cost curve | 03, 04 | 3–5 min | `models/meta.json`, `models/ridge_h*.json`, `models/event_h*.txt`, `app/data/*.csv.gz`, `cost_curve.csv` |
| 07 | `07_live_model.py` | Live fallback model that needs no measured PM2.5 (weather + CAMS from Open-Meteo): backtest, export, one real live forecast | 06 | 8–12 min | `live_model_*.csv`, `models/meta_live.json`, `models/live_*.txt` |
| 08 | `08_live_network.py` | Live low-cost sensor network on OpenAQ: downloads history, chooses the 15 sensors the app reads, trains Ridge models for every horizon 1–72 h | sensor-check cells, OpenAQ key | ~10 min | `app/data/live_sensors.csv`, `models/hourly_*.json`, `models/meta_hourly.json`, `network_calibration.json` |
| 09 | `09_proposal_checks.py` | Remaining proposal items: recall at fixed precision, fold mean ± std, false alarms per month (days and episodes), seaborn heatmap, autocorrelation, search for other stations | 03, 04 (part E needs the OpenAQ key) | 1 min + 5–15 min | `rq3_recall_at_precision.csv`, `classification_fold_stats.csv`, `false_alarms_per_month.csv`, `multistation_candidates.csv`, `multistation_summary.json` |
| 10 | `10_clarity_bridge.py` | Bridge calibration of the live network to the Embassy scale through a Clarity sensor; second-station gap filling tested in the backtest | 08, OpenAQ key | ~10 min | `bridge_calibration.json`, `multistation_backtest.csv`, `multistation_dm.csv`, `multistation_folds.csv`, updates `models/meta_hourly.json` |
| 11 | `11_rq2_same_period.py` | RQ2 re-check: random split and rolling origin scored on exactly the same test hours | 03 | ~2 min | `rq2_same_period.csv` |
| 12 | `12_cost_sensitive.py` | Cost-sensitive regression (instructor's suggestion): peak-hour and smog-season sample weights, quantile-loss LightGBM, rule fixed in advance on pre-test data | 03 | 10–15 min | `cost_sensitive_*.csv`, `cost_sensitive_settings.json` |

Notes:

- Notebook 08 needs `data/raw/openaq_lahore_active_sensors.csv`. This file is written by the sensor-check cells
  (cells 19 and 20 of `PM25_Main.ipynb`), which list all OpenAQ PM2.5 sensors within 25 km of Lahore
  and how many hours each reported in the last 72 hours.
- Notebook 08 writes `results/network_pipeline_test.csv` and `results/figures/network_vs_embassy.png` only if
  the live sensors overlap the Embassy monitor by at least 300 hours. They do not overlap at all, so these two
  files do not exist. That is why notebook 10 calibrates the network through the Clarity bridge sensor.
- Notebooks 09 and 10 both write `multistation_backtest.csv` and `multistation_dm.csv`. The committed versions
  come from notebook 10 (run later), which uses the Clarity sensor.

---

## 6. The Colab notebook `PM25_Main.ipynb`, cell by cell

[`Colab Code file/PM25_Main.ipynb`](Colab%20Code%20file/PM25_Main.ipynb) is the notebook we ran in Colab.
It calls the scripts in `notebooks/` and keeps their printed output, so it is the run log of the project.

| Cell | What it does |
|---|---|
| 0 | Title, project and team |
| 1 | Settings: GitHub user, repository name, git name and email, Drive folder |
| 2 | Mounts Google Drive, clones the repository into Drive (first time) and moves into it (needs `GITHUB_TOKEN`) |
| 3 | Makes the OpenAQ key available from Colab secrets and installs `requirements.txt` |
| 4 | Helper functions `pull()` and `push(message)`; the token is masked in any printed output |
| 5 | Runs **01** data acquisition |
| 6–7 | Pulls the latest code and lists `src/` and `notebooks/` |
| 8 | Runs **02** cleaning, features and EDA |
| 9 | Runs **03** regression backtest (RQ1, RQ2) |
| 10 | First run of **04**; superseded by cell 13 after the threshold grid was refined |
| 11 | Pushes the results of step 3 to GitHub |
| 12 | Checks the size of the large (git-ignored) data files |
| 13 | Final run of **04** classification (RQ3) |
| 14 | Runs **05** LSTM and SHAP |
| 15 | Pushes the results of step 4 |
| 16 | Runs **06** model export |
| 17 | Runs **07** live fallback model |
| 18 | Pushes the exported models and dashboard data |
| 19 | Sensor check: lists OpenAQ PM2.5 sensors within 25 km. The red error is an HTTP 500 from the OpenAQ server for one sensor; cell 20 repeats the step with retries |
| 20 | Sensor check with retries; saves `data/raw/openaq_lahore_active_sensors.csv` (needed by 08) |
| 21 | Runs **08** live sensor network and hourly models |
| 22 | Pushes the live network results |
| 23 | Runs **09** proposal checks |
| 24 | Pushes the proposal-check results |
| 25 | Runs **10** Clarity bridge calibration and second-station test |
| 26 | Pushes the bridge results |
| 27 | Pulls the latest code |
| 28 | Security check: lists tracked `.env` files (only `.env.example`) and scans the git history for leaked tokens or keys (none found) |
| 29 | Runs **11** RQ2 same-period check |
| 30 | Pushes the RQ2 results |
| 31 | Runs **12** cost-sensitive regression |
| 32 | Pushes the cost-sensitive results |

---

## 7. Results folder guide

All tables are in [`results/`](results). Each file is written by the notebook shown.

**Data and EDA**

| File | Notebook | Contents |
|---|---|---|
| `data_summary.json` | 02 | Hours, missing data, cleaning counts, event rates by season, CAMS vs monitor (r, bias, MAE) |

**PM2.5 forecasting (RQ1)**

| File | Notebook | Contents |
|---|---|---|
| `regression_summary.csv` | 03 | MAE, RMSE, R², MAPE, skill vs persistence and vs CAMS, fold mean/std, for every model and horizon |
| `regression_by_fold.csv` | 03 | The same metrics for each of the 10 test folds |
| `regression_by_season.csv` | 03 | MAE in the smog season (Oct–Feb) and the rest of the year |
| `dm_tests.csv` | 03 | Diebold-Mariano tests: tuned LightGBM + CAMS against every other model |
| `bootstrap_ci.csv` | 03 | Day-block bootstrap 95% intervals for MAE differences |
| `optuna_best_params.json` | 03 | Best LightGBM settings from Optuna and the validation MAE before and after tuning |
| `lstm_comparison.csv` | 05 | LSTM against Ridge, persistence and CAMS on the same test hours, overall and by season |
| `lstm_dm_tests.csv` | 05 | Diebold-Mariano tests for the LSTM |
| `shap_importance.csv` | 05 | Mean absolute SHAP value of every input at 24, 48 and 72 h |

**Random split vs rolling origin (RQ2)**

| File | Notebook | Contents |
|---|---|---|
| `rq2_random_vs_rolling.csv` | 03 | Random 80/20 split (all hours since Aug 2022) vs rolling origin |
| `rq2_same_period.csv` | 11 | Random split vs rolling origin scored on exactly the same test hours (used in the paper) |

**Severe-smog warnings (RQ3)**

| File | Notebook | Contents |
|---|---|---|
| `classification_summary.csv` | 04 | Recall, precision, F1, PR-AUC, ROC-AUC, cost and threshold for each model, horizon, event level and cost ratio |
| `classification_by_season.csv` | 04 | The same metrics split by season (Oct–Feb vs Mar–Sep) |
| `lead_time.csv` | 04 | Onsets detected and how many hours in advance (after 24 or 72 clean hours) |
| `cost_curve.csv` | 06 | Expected cost against the decision threshold |
| `rq3_recall_at_precision.csv` | 09 | Recall at a fixed precision of at least 0.80, 0.90 and 0.95 |
| `classification_fold_stats.csv` | 09 | Recall, precision and F1 as mean ± std over folds |
| `false_alarms_per_month.csv` | 09 | False alarms per month, counted as days and as episodes |

**Cost-sensitive regression (instructor's suggestion)**

| File | Notebook | Contents |
|---|---|---|
| `cost_sensitive_tuning.csv` | 12 | Every weighting setting tried on the pre-test validation blocks |
| `cost_sensitive_settings.json` | 12 | The settings chosen by the fixed rule |
| `cost_sensitive_summary.csv` | 12 | Overall MAE, MAE on high and clean hours, bias on Hazardous hours |
| `cost_sensitive_by_band.csv` | 12 | MAE by observed PM2.5 band |
| `cost_sensitive_dm.csv`, `cost_sensitive_bootstrap.csv` | 12 | Significance tests against the unweighted models |

**Live system and second station**

| File | Notebook | Contents |
|---|---|---|
| `live_model_regression.csv` | 07 | Backtest of the weather + CAMS live model against persistence, CAMS and the full model |
| `live_model_events.csv` | 07 | Live warning model: recall and precision |
| `live_model_dm_tests.csv` | 07 | Diebold-Mariano tests for the live model |
| `network_calibration.json` | 08 | Direct calibration attempt (no overlap with the Embassy monitor, so identity) |
| `bridge_calibration.json` | 10 | The two bridge fits through the Clarity sensor and the combined calibration |
| `multistation_candidates.csv`, `multistation_summary.json` | 09 | Other OpenAQ sensors that ran alongside the Embassy monitor |
| `multistation_backtest.csv`, `multistation_dm.csv`, `multistation_folds.csv` | 10 | Backtest with second-station gap filling, its tests and the hours filled per fold |

---

## 8. Figures folder guide

All plots are in [`results/figures/`](results/figures).

| Figure | Notebook | What it shows |
|---|---|---|
| `monthly_mean.png` | 02 | Mean PM2.5 by month |
| `monthly_boxplot.png` | 02 | Distribution of PM2.5 by month |
| `diurnal_cycle.png` | 02 | Mean PM2.5 by hour of day, smog season vs rest of year |
| `weather_correlation.png` | 02 | Spearman correlation of PM2.5 with the weather variables |
| `cams_vs_observed.png` | 02 | CAMS forecast against the monitor (shows the strong low bias) |
| `heatmap_month_hour.png` | 09 | Month × hour heatmap of mean PM2.5 (seaborn) |
| `acf_pm25.png` | 09 | Autocorrelation of PM2.5 (statsmodels), the basis for lags 1, 24 and 48 h |
| `mae_by_horizon.png` | 03 | MAE of every model at 24, 48 and 72 h |
| `mae_by_fold.png` | 03 | MAE at 24 h in each of the 10 test folds (seasonal variation) |
| `example_forecast.png` | 03 | 24-h forecasts vs observations, November 2024 |
| `lstm_comparison.png` | 05 | LSTM against Ridge, persistence and CAMS |
| `shap_summary_h48.png` | 05 | SHAP summary (beeswarm) for LightGBM + CAMS at 48 h |
| `shap_importance_by_horizon.png` | 05 | How input importance changes from 24 to 72 h |
| `shap_dependence_cams_h48.png` | 05 | How the CAMS input changes the forecast |
| `pr_curves_h48.png` | 04 | Precision-recall curves of the warning models at 48 h |
| `recall_by_season.png` | 04 | Warning recall by horizon and season (10:1 cost) |
| `cost_curve.png` | 06 | Expected cost against the decision threshold |
| `live_model_backtest.png` | 07 | Live (weather + CAMS) model against the baselines and the full model |
| `bridge_calibration.png` | 10 | The two legs of the bridge calibration through the Clarity sensor |
| `cost_sensitive_bands.png` | 12 | Error by PM2.5 band, unweighted vs peak-weighted vs quantile models |
| `cost_sensitive_example.png` | 12 | One month of 24-h forecasts: Ridge vs strongly peak-weighted Ridge |

---

## 9. Where each table and figure of the paper comes from

| In the paper | Source in this repository |
|---|---|
| Table I, data summary | `results/data_summary.json` |
| Table II, regression results | `results/regression_summary.csv`, `results/lstm_comparison.csv` |
| Table III, Diebold-Mariano tests | `results/dm_tests.csv`, `results/lstm_dm_tests.csv` |
| Table IV, RQ2 random vs rolling | `results/rq2_same_period.csv` |
| Table V, severe-smog warnings | `results/classification_summary.csv`, `results/rq3_recall_at_precision.csv`, `results/false_alarms_per_month.csv` |
| Table VI, lead time | `results/lead_time.csv` |
| Table VII, cost-sensitive regression | `results/cost_sensitive_summary.csv` |
| Table VIII, live model | `results/live_model_regression.csv`, `results/live_model_events.csv` |
| Fig. 1, month × hour heatmap | `results/figures/heatmap_month_hour.png` |
| Fig. 2, CAMS vs monitor | `results/figures/cams_vs_observed.png` |
| Fig. 3, workflow | drawn in the paper (Section 3 of this README) |
| Fig. 4, MAE by horizon | redrawn from `regression_summary.csv`, `lstm_comparison.csv`, `live_model_regression.csv` (see also `mae_by_horizon.png`) |
| Fig. 5, November 2024 forecasts | `results/figures/example_forecast.png` |
| Fig. 6, SHAP by horizon | redrawn from `results/shap_importance.csv` (see also `shap_importance_by_horizon.png`) |
| Fig. 7, precision-recall curves | `results/figures/pr_curves_h48.png` |
| Fig. 8, cost-sensitive error by band | redrawn from `results/cost_sensitive_by_band.csv` (see also `cost_sensitive_bands.png`) |
| Fig. 9, bridge calibration | `results/figures/bridge_calibration.png` |

---

## 10. Models, app data and the data folder

**`models/`** holds the exported final models. We use no pickle files, so loading a model cannot run hidden code.

| File | Contents |
|---|---|
| `meta.json` | Full model: which file and which inputs to use for each horizon; warning threshold |
| `ridge_h24.json`, `ridge_h48.json`, `ridge_h72.json` | Ridge + CAMS PM2.5 models (inputs, imputation medians, scaling, coefficients) |
| `event_h24.txt`, `event_h48.txt`, `event_h72.txt` | LightGBM + SMOTE severe-smog warning models (LightGBM text format) |
| `meta_live.json`, `live_pm25_h*.txt`, `live_event_h*.txt` | Fallback live model (weather + CAMS only) |
| `meta_hourly.json`, `hourly_full.json`, `hourly_live.json` | Ridge models for every horizon 1–72 h and the bridge calibration, used for the hour-by-hour live chart |

**`app/data/`** holds small snapshots for the dashboard: `backtest.csv.gz` (out-of-sample backtest forecasts),
`observed.csv.gz` (observed PM2.5), `recent_features.csv.gz` (recent model inputs for the forecast explorer and
the API) and `live_sensors.csv` (the 15 sensors the live tab reads).

**`data/`** is not committed (see `.gitignore`), because it is large and can be re-created. Notebook 01 downloads
`data/raw/`, notebook 02 writes `data/features.csv`, and notebooks 03, 04 and 12 write the large prediction
files `data/predictions_*.csv`.

---

## 11. Live forecasting system

The US Embassy monitor's public feed ended in March 2025. About 50 low-cost PM2.5 sensors in Lahore still report
to OpenAQ (AirGradient and Hawanama networks). The live tab and the API's `/live` endpoint:

1. read the 15 AirGradient sensors listed in `app/data/live_sensors.csv` and take their hourly median;
2. convert the median to the Embassy (training) scale. None of these sensors ran while the Embassy monitor did,
   so a Clarity sensor (OpenAQ 7466365) that overlapped both is used as a bridge:
   Embassy vs Clarity (Nov 2023 to Feb 2025, r = 0.91) and Clarity vs network median (May 2025 to Sep 2026,
   r = 0.95). Combined: **Embassy scale = 17.68 + 0.9367 × network median**;
3. combine this with the Open-Meteo weather forecast and the CAMS forecast, and run the full model for an
   hour-by-hour 72-hour forecast and smog warnings. The current reading is shown in µg/m³ and as US AQI
   (EPA 2024 breakpoints);
4. check the last 7 days of 24-hour forecasts against the calibrated network.

If the sensors cannot be read, the app switches to the fallback weather + CAMS model from notebook 07 and
says so on screen.

---

## 12. Dashboard and API

**Dashboard tabs** (`app/streamlit_app.py`):

| Tab | Contents |
|---|---|
| Live forecast | Current PM2.5 and US AQI, 72-hour forecast, smog warnings, 7-day verification |
| Forecast explorer | Backtest forecasts against observations, and smog warnings, for a chosen month of the test period |
| Model results | RQ1 tables and charts, Diebold-Mariano and bootstrap tests, RQ2, RQ3, combined stations and the live model |
| Explainability | SHAP results |
| Deployed model & API | How the deployed models and the API work |
| About | Project summary, data sources and limitations |

**API** (`api/main.py`, interactive documentation at `/docs`):

| Endpoint | Description |
|---|---|
| `GET /health` | Status, training period and data snapshot range |
| `GET /live` | Today's forecast from the live sensor network (or the weather + CAMS fallback) |
| `GET /forecast?time=2025-03-01T08:00` | Forecast for an hour inside the stored data snapshot |
| `GET /features/example` | A complete example input row |
| `POST /predict` | Forecast from your own inputs: `{"issued_at": "...", "features": {...}}` |

---

## 13. Security of API keys

- The OpenAQ key and the GitHub token are never written in code or committed.
- In Colab they are read from **Colab Secrets**, and on Streamlit Cloud from **Streamlit secrets**.
  Locally they are read from a `.env` file, which `.gitignore` excludes. `.env.example` is only a template.
- `src/config.py` stops with a clear message if a key is missing. Git output in Colab masks the token.
- Cell 28 of `PM25_Main.ipynb` checked that no `.env` file is tracked and that no token or key appears
  anywhere in the git history.

---

## 14. Data sources

- **PM2.5 (training and backtest):** US Diplomatic Post Lahore, through [OpenAQ](https://openaq.org) (sensor 25135),
  hourly, May 2019 to March 2025.
- **PM2.5 (live):** 15 AirGradient low-cost sensors through OpenAQ; Clarity sensor 7466365 as the calibration bridge.
- **Weather:** ERA5 reanalysis through the [Open-Meteo](https://open-meteo.com) historical weather API; the live
  system uses the Open-Meteo weather forecast.
- **CAMS:** Copernicus Atmosphere Monitoring Service global PM2.5 forecast through the Open-Meteo Air Quality API
  (from August 2022).

This work was generated using Copernicus Atmosphere Monitoring Service information and contains modified
Copernicus Climate Change Service information.

---

## 15. Limitations

- Training and evaluation use a single reference monitor.
- Archived CAMS values may not match the lead time of a real-time CAMS forecast, so CAMS may help slightly more
  in the backtest than in live use.
- Live readings are calibrated only indirectly, through one Clarity sensor that is assumed not to drift between
  the two calibration periods. Live weather comes from a forecast model rather than ERA5.
- Tree-model probabilities are poorly calibrated, so the cost-optimal thresholds are very low and the warnings
  give more false alarms than persistence.
- Only 7 to 16 smog-episode onsets fall in the test period, too few for firm lead-time conclusions.
- All models under-predict the highest night-time peaks. Sample weighting did not fix this. Fire (crop-residue
  burning) data and probabilistic forecasts are planned as future work.

---

*Acknowledgment: we thank Dr Mateen Yaqoob for his guidance in the DSTT course and for suggesting the
peak-hour weighting tested in notebook 12.*
