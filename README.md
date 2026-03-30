# G. Setup Instructions

---

## Prerequisites

Before cloning the repository, ensure the following are installed and meet the minimum version requirements:

| Dependency | Minimum Version | Notes |
| :--- | :--- | :--- |
| **Python** | 3.9+ | [Download](https://www.python.org/downloads/) |
| **PostgreSQL** | 14+ | [Download](https://www.postgresql.org/download/) |
| **Docker & Docker Compose** | Docker 24+ / Compose v2+ | Required to run the PostgreSQL container |
| **MLflow** | Installed via `requirements.txt` | Run `mlflow ui` separately to view experiment results |

> **Visual Crossing API Key:** You will need a free API key from [Visual Crossing](https://www.visualcrossing.com/sign-up). After signing up, your key is available under **Account > Key**.

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

> **Note:** `requirements.txt` includes `python-dotenv`, which is required for loading environment variables from the `.env` file. If you encounter `ModuleNotFoundError: No module named 'dotenv'`, run `pip install python-dotenv` manually.

---

## Step 4 — Configure Environment Variables

Create a `.env` file in the **root** of the project directory with the following contents:

```env
DB_USER=your_db_username
DB_PASS=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=abuja_air_quality
VISUAL_CROSSING_API_KEY=your_api_key_here
MLFLOW_TRACKING_URI=./mlruns
```

> **Tip:** Replace `your_db_username` and `your_db_password` with the credentials configured in `docker-compose.yml`. Do not commit the `.env` file to version control — it is already listed in `.gitignore`.

---

## Step 5 — Start the PostgreSQL Container

The project uses Docker Compose to run a managed PostgreSQL instance. From the root directory, run:

```bash
docker compose up -d
```

**Expected output:**
```
✔ Container air_quality_db       Started                                                              
✔ Container air_quality_pgadmin  Started
```

To confirm the container is running:
```bash
docker ps
```
You should see a container named `mlops_weather_pipeline-db-1` (or similar) with status `Up`.

---

## Step 6 — Initialize the Database and Load Historical Data

```bash
# Create the database schema (tables, indexes, composite primary keys)
python data_ingestion/setup_db.py

# Seed the database with the historical CSV dataset
python data_ingestion/load_history.py
```

**Expected output from `load_history.py`:**
```
[Verification] Total rows in 'daily_weather': ****
```

Then fetch the first batch of live data from the Visual Crossing API:

```bash
python data_ingestion/ingest.py
```

**Expected output:**
```
Fetched data for YYYY-MM-DD. Upserted N records.
```

> **API Troubleshooting:** If you see a `401 Unauthorized` error, double-check that your `VISUAL_CROSSING_API_KEY` in `.env` is correct and has not exceeded the free tier's daily call limit (1,000 records/day).

---

## Step 7 — Run the Validation Script (Parameter Selection)

The `validate.py` script runs Auto-ARIMA to identify the optimal `(p, d, q)` parameters and performs Walk-Forward Cross-Validation to calculate a realistic MAE before any model is trained.

```bash
python validate.py
```

**Expected output:**
```
Best ARIMA parameters: (p=..., d=..., q=...)
Walk-Forward MAE: X.XX °C
Model passed quality gate. Safe to proceed with training.
or
Deployment Halted. Recent MAE (***) > current.
```

> If MAE exceeds the defined threshold, training will be blocked. This is expected behaviour — it means the quality gate is working. Check data completeness and re-run `ingest.py` if needed.

---

## Step 8 — Train the Model

The `train.py` script uses the parameters surfaced by `validate.py`, trains the SARIMA model, and registers it in the MLflow Model Registry.

```bash
python train.py
```

**Expected output:**
```
Training model with params (p=..., d=..., q=...)...
Logged run to MLflow: run_id=XXXXXXXXXXXXXX
Registered model version: v1
```

### Viewing Experiment Results in MLflow (Optional but Recommended)

To explore training metrics, parameters, and registered models visually:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser. You will see a logged run under the experiment name defined in `train.py`.

---

## Step 9 — Generate Predictions

The `predict.py` script loads the trained model from the MLflow registry, fetches the latest "gap data" from the database, updates the model state, and writes the 3-day forecast back to PostgreSQL.

```bash
python deployment/predict.py
```

**Expected output:**
```
Loaded model version v1 from MLflow registry.
Fetched N days of gap data from DB.
Forecast written to database:
  Day 1 (YYYY-MM-DD): XX.X °C
  Day 2 (YYYY-MM-DD): XX.X °C
  Day 3 (YYYY-MM-DD): XX.X °C
```

---

## Step 10 — Launch the Dashboard

```bash
streamlit run deployment/app.py
```

Streamlit will automatically open the dashboard in your browser at [http://localhost:8501](http://localhost:8501). The dashboard reads forecast and actuals data directly from PostgreSQL and renders an interactive Plotly chart.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| `could not connect to server` (PostgreSQL) | Container not running | Run `docker compose up -d` and re-check with `docker ps` |
| `ModuleNotFoundError: No module named 'dotenv'` | Missing dependency | Run `pip install python-dotenv` |
| `401 Unauthorized` (Visual Crossing) | Bad or missing API key | Verify `VISUAL_CROSSING_API_KEY` in `.env` |
| `Model not found` in `predict.py` | Training not yet run | Complete Step 8 before Step 9 |
| `MAE threshold exceeded` in `validate.py` | Insufficient or stale data | Re-run `ingest.py` to fetch fresh data, then retry |
| MLflow UI shows no experiments | Wrong tracking URI | Ensure `MLFLOW_TRACKING_URI=./mlruns` is set in `.env` and you are running `mlflow ui` from the project root |
| Streamlit dashboard shows no forecast | `predict.py` not yet run | Complete Step 9 before launching the dashboard |
