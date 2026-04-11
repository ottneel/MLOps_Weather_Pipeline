import numpy as np
import pandas as pd
import urllib.parse
import os
from sqlalchemy import create_engine

def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{urllib.parse.quote_plus(os.getenv('DB_PASS'))}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

def load_and_engineer(city="Abuja"):
    """
    Single source of truth for data loading and feature engineering.
    Used by validate.py, train.py and predict.py.
    Any change to features only needs to happen here.
    """
    engine = get_db_engine()
    df = pd.read_sql(
        "SELECT * FROM daily_weather WHERE city = city ORDER BY date",
        engine, params={'city': city},
        index_col='date', parse_dates=['date']
    )

    df['pressure']   = df['pressure'].ffill()
    df['visibility'] = df['visibility'].ffill()
    df['is_rain']    = (df['precip'] > 0).astype(int)

    to_drop = [
        'precip', 'precipprob', 'precipcover', 'preciptype',
        'conditions', 'description', 'icon', 'severerisk',
        'snow', 'snowdepth', 'feelslike', 'feelslikemax', 'feelslikemin',
        'dew', 'uvindex', 'temp_max', 'temp_min', 'solarenergy',
        'city', 'source', 'stations', 'sunrise', 'sunset', 'windgust'
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

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

    for col in ['humidity', 'cloudcover', 'visibility']:
        df[f'{col}_delta'] = df[f'{col}_lag1'] - df[f'{col}_lag2']
    df['pressure_delta'] = df['pressure'] - df['pressure_lag1']
    df['temp_avg_delta'] = df['temp_avg'] - df['temp_avg_lag1']

    df['month_sin']   = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos']   = np.cos(2 * np.pi * df.index.month / 12)
    df['winddir_sin'] = np.sin(2 * np.pi * df['winddir'] / 360)
    df['winddir_cos'] = np.cos(2 * np.pi * df['winddir'] / 360)
    df = df.drop(columns=['winddir'])

    for col in ['pressure', 'humidity']:
        df[f'{col}_roll3'] = df[col].shift(1).rolling(3).mean()
        df[f'{col}_roll7'] = df[col].shift(1).rolling(7).mean()

    t0_to_drop = ['temp_avg', 'humidity', 'pressure', 'cloudcover',
                  'visibility', 'windspeed', 'solarradiation', 'moonphase']
    df = df.drop(columns=[c for c in t0_to_drop if c in df.columns])

    return df.dropna()