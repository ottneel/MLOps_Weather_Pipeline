import os
import numpy as np
import pandas as pd
import urllib.parse
import mlflow
import mlflow.xgboost
from sqlalchemy import create_engine, text
from datetime import datetime
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
        LIMIT 14
    """)
    df = pd.read_sql(query, engine, params={'city': CITY},
                     index_col='date', parse_dates=['date'])

    # Sort ascending so lags are computed in the right direction
    return df.sort_index()

# ── BUILD FEATURES ────────────────────────────────────────────────────────────
def build_features(df):
    """
    Replicates the exact same feature engineering from rain_features_eng.py.
    The last row of the output is today's feature vector.
    """
    df['pressure']   = df['pressure'].ffill()
    df['visibility'] = df['visibility'].ffill()

    # Lags
    lag_config = {
        'pressure':   [1],
        'humidity':   [1, 2],
        'temp_avg':   [1],
        'cloudcover': [1, 2],
        'visibility': [1, 2, 3],
    }
    for col, lags in lag_config.items():
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Deltas
    for col in ['humidity', 'cloudcover', 'visibility']:
        df[f'{col}_delta'] = df[f'{col}_lag1'] - df[f'{col}_lag2']
    df['pressure_delta'] = df['pressure'] - df['pressure_lag1']
    df['temp_avg_delta'] = df['temp_avg'] - df['temp_avg_lag1']

    # Cyclical encoding
    df['month_sin']   = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos']   = np.cos(2 * np.pi * df.index.month / 12)
    df['winddir_sin'] = np.sin(2 * np.pi * df['winddir'] / 360)
    df['winddir_cos'] = np.cos(2 * np.pi * df['winddir'] / 360)
    df = df.drop(columns=['winddir'])

    # Rolling means (window ends at T-1)
    for col in ['pressure', 'humidity']:
        df[f'{col}_roll3'] = df[col].shift(1).rolling(3).mean()
        df[f'{col}_roll7'] = df[col].shift(1).rolling(7).mean()

    # Drop T0 raw features — can't use today to predict today
    t0_to_drop = ['temp_avg', 'humidity', 'pressure', 'cloudcover',
                  'visibility', 'windspeed', 'solarradiation', 'moonphase']
    df = df.drop(columns=[c for c in t0_to_drop if c in df.columns])

    return df

# ── SAVE FORECAST TO DB ───────────────────────────────────────────────────────
def save_forecast(engine, forecast_date, prediction, probability):
    row = pd.DataFrame([{
        'forecast_date':     forecast_date,
        'city':              CITY,
        'predicted_rain':    int(prediction),
        'rain_probability':  round(float(probability), 4),
        'model_version':     'AbujaRain_Production',
        'created_at':        datetime.now()
    }])
    row.to_sql('daily_rain_forecasts', engine, if_exists='append', index=False)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def predict():
    engine = get_db_engine()

    # 1. Load production model from MLflow
    print(f"Loading production model '{MODEL_NAME}'...")
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/Production")

    # 2. Load recent raw data
    df = load_recent_data(engine)
    print(f"Loaded {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")

    # 3. Build features
    df = build_features(df)

    # 4. Take the last row — today's feature vector
    today      = df.index[-1].date()
    X_today    = df.iloc[[-1]].dropna()

    X_today = df.iloc[[-1]]
    print("NaN features:", X_today.columns[X_today.isna().any()].tolist())
    X_today = X_today.dropna()
    
    if X_today.empty:
        print("ERROR: Not enough data to build features. Need at least 7 days in DB.")
        return

    # 5. Predict
    prediction  = model.predict(X_today)[0]
    probability = model.predict_proba(X_today)[0][1]  # P(rain)

    print(f"\nForecast for {today}:")
    print(f"  Prediction:  {'Rain' if prediction == 1 else 'No Rain'}")
    print(f"  Confidence:  {probability:.1%}")

    # 6. Save to DB
    save_forecast(engine, today, prediction, probability)
    print("Forecast saved to DB.")

if __name__ == "__main__":
    predict()