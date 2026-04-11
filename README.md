# Abuja Weather Forecast — MLOps Pipeline

A production-style MLOps pipeline that delivers daily temperature forecasts and rain predictions for Abuja, Nigeria. Built with SARIMA for temperature and XGBoost for rainfall classification.

---

## What This Project Does

- Ingests daily weather data from the Visual Crossing API into PostgreSQL
- Predicts **tomorrow's temperature** using a SARIMA time series model
- Predicts **whether it will rain tomorrow** using an XGBoost classifier
- Serves both forecasts on a Streamlit dashboard
- Tracks all experiments, parameters, and model versions with MLflow

---

## Project Structure

```
MLOps_Weather_Pipeline/
│
├── data_ingestion/
│ ├── setup_db.py # Creates all database tables
│ ├── load_history.py # Seeds DB from historical CSV
│ └── fetch_daily.py # Fetches yesterday's data from API daily
│
├── temperature/
│ ├── validate.py # Auto-ARIMA search + walk-forward validation
│ ├── train.py # Trains SARIMA on full data, registers in MLflow
│ └── predict.py # Generates 3-day temperature forecast
│
├── rainfall/
│ ├── rain_features_eng.py # Single source of truth for feature engineering
│ ├── rain_validate.py # GridSearch + walk-forward validation
│ ├── rain_train.py # Trains XGBoost on full data, registers in MLflow
│ └── rain_predict.py # Generates tomorrow's rain prediction
│
├── app.py # Streamlit dashboard
├── docker-compose.yml # PostgreSQL container
├── requirements.txt
└── .env # Not committed — see Step 4
```

---

## Prerequisites

| Dependency | Minimum Version | Notes |
| :--- | :--- | :--- |
| Python | 3.9+ | [Download](https://www.python.org/downloads/) |
| PostgreSQL | 14+ | Runs via Docker |
| Docker & Docker Compose | Docker 24+ / Compose v2+ | Required for the database container |
| Visual Crossing API Key | Free tier | [Sign up here](https://www.visualcrossing.com/sign-up) |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/ottneel/MLOps_Weather_Pipeline.git
cd MLOps_Weather_Pipeline
```

---

## Step 2 — Set Up Python Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Configure Environment Variables

Create a `.env` file in the root of the project:

```env
DB_USER=your_db_username
DB_PASS=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=abuja_weather
VISUAL_CROSSING_API_KEY=your_api_key_here
MLFLOW_TRACKING_URI=./mlruns
```

> The `.env` file is listed in `.gitignore` and will never be committed. Never share your API key or database credentials publicly.

---

## Step 5 — Start the PostgreSQL Container

```bash
docker compose up -d
```

Confirm it is running:

```bash
docker ps
```

You should see the database container with status `Up`.

---

## Step 6 — Initialise the Database and Load Historical Data

```bash
# Create all tables and indexes
python data_ingestion/setup_db.py

# Load historical CSV data into the database
python data_ingestion/load_history.py

# Fetch the most recent data from the Visual Crossing API
python data_ingestion/fetch_daily.py
```

After `load_history.py` you should see:

```
[Verification] Total rows in 'daily_weather': XXXX
```

---

## Step 7 — Temperature Pipeline

Run these three scripts in order:

```bash
# 1. Find best ARIMA parameters via Auto-ARIMA and walk-forward validation
python temperature/validate.py

# 2. Train on full dataset and register model in MLflow
python temperature/train.py

# 3. Generate 3-day temperature forecast and save to database
python temperature/predict.py
```

Expected output from `predict.py`:

```
Forecast written to database:
Day 1 (YYYY-MM-DD): XX.X°C
Day 2 (YYYY-MM-DD): XX.X°C
Day 3 (YYYY-MM-DD): XX.X°C
```

---

## Step 8 — Rainfall Pipeline

Run these three scripts in order:

```bash
# 1. Tune XGBoost hyperparameters via GridSearch + walk-forward validation
python rainfall/rain_validate.py

# 2. Train on full dataset and register model in MLflow
python rainfall/rain_train.py

# 3. Generate tomorrow's rain prediction and save to database
python rainfall/rain_predict.py
```

Expected output from `rain_predict.py`:

```
Forecast for YYYY-MM-DD:
Prediction: Rain / No Rain
Confidence: XX.X%
Forecast saved to DB.
```

---

## Step 9 — View Experiments in MLflow (Optional)

```bash
python -m mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) to explore logged parameters, metrics, confusion matrix images, and registered model versions for both pipelines.

---

## Step 10 — Launch the Dashboard

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The dashboard displays:

- Tomorrow's rain prediction with probability score
- Rain probability trend over recent days
- Latest temperature reading and 3-day forecast
- Temperature history vs forecast chart

---

## Daily Operations

Once the pipeline is set up, the daily routine is:

```bash
python data_ingestion/ingest.py # pull yesterday's actuals
python temperature/predict.py # generate temperature forecast
python rainfall/rain_predict.py # generate rain forecast
```

Retraining is recommended weekly:

```bash
python temperature/validate.py && python temperature/train.py
python rainfall/rain_validate.py && python rainfall/rain_train.py
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| `could not connect to server` | Container not running | Run `docker compose up -d` |
| `ModuleNotFoundError` | Missing dependency | Run `pip install -r requirements.txt` |
| `401 Unauthorized` (Visual Crossing) | Bad API key | Check `VISUAL_CROSSING_API_KEY` in `.env` |
| `No validation experiment found` | validate.py not run yet | Run validate.py before train.py |
| `feature_names mismatch` | Model trained with different features | Re-run validate.py and train.py |
| `Model not found in MLflow` | Training not completed | Run the full validate → train sequence |
| `No data in dashboard` | predict.py not run | Run both predict scripts first |
| MLflow UI shows no experiments | Wrong tracking URI | Set `MLFLOW_TRACKING_URI=./mlruns` in `.env` |
| `streamlit: command not found` | PATH issue | Use `python -m streamlit run app.py` |
