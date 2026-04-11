import os
import urllib.parse
import mlflow
import mlflow.xgboost
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from rain_features_eng import engineer_features
from dotenv import load_dotenv

# ── SETUP ─────────────────────────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path, override=True)

CITY       = "Abuja"
MODEL_NAME = "AbujaRain"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))

# ── DATABASE ──────────────────────────────────────────────────────────────────
def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{urllib.parse.quote_plus(os.getenv('DB_PASS'))}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

# ── LOAD RECENT RAW DATA ──────────────────────────────────────────────────────
def load_recent_data(engine):
    """
    Fetch the last 10 days from DB.
    We need at least 7 days to build all lags and rolling means.
    """
    query = text("""
        SELECT date, temp_avg, humidity, pressure, cloudcover,
               visibility, windspeed, winddir, solarradiation, moonphase
        FROM daily_weather
        WHERE city = :city
        ORDER BY date DESC
        LIMIT 10
    """)
    df = pd.read_sql(query, engine, params={'city': CITY},
                     index_col='date', parse_dates=['date'])

    # Sort ascending so lags are computed in the right direction
    return df.sort_index()

# ── SAVE FORECAST TO DB ───────────────────────────────────────────────────────
def save_forecast(engine, forecast_date, prediction, probability):
    row = pd.DataFrame([{
        'forecast_date':    forecast_date,
        'city':             CITY,
        'predicted_rain':   int(prediction),
        'rain_probability': round(float(probability), 4),
        'model_version':    'AbujaRain_Production',
        'created_at':       datetime.now()
    }])
    row.to_sql('daily_rain_forecasts', engine, if_exists='append', index=False)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def predict():
    engine = get_db_engine()

    # 1. Load production model from MLflow
    print(f"Loading production model '{MODEL_NAME}'...")
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/Production")

    # 2. Load last 10 days of raw data
    df = load_recent_data(engine)
    print(f"Loaded {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")

    # 3. Engineer features using the same function as training
    df = engineer_features(df)

    # 4. Take the last row — today's feature vector
    today   = df.index[-1].date()
    target  = today + timedelta(days=1)
    X_today = df.iloc[[-1]].dropna()

    if X_today.empty:
        print("ERROR: Not enough data to build features. Need at least 7 days in DB.")
        return

    # Reorder columns to exactly match what the model was trained on
    X_today = X_today[model.feature_names_in_]

    # 5. Predict
    prediction  = model.predict(X_today)[0]
    probability = model.predict_proba(X_today)[0][1]  # P(rain)

    print(f"\nForecast for {target}:")
    print(f"  Prediction:  {'Rain' if prediction == 1 else 'No Rain'}")
    print(f"  Confidence:  {probability:.1%}")

    # 6. Save to DB
    save_forecast(engine, target, prediction, probability)
    print("Forecast saved to DB.")

if __name__ == "__main__":
    predict()