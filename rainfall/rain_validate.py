import os
import json
import numpy as np
import pandas as pd
import urllib.parse
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sqlalchemy import create_engine
from dotenv import load_dotenv


# Initial Setup
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path, override=True)

CITY            = "Abuja"
EXPERIMENT_NAME = "Abuja_Rain_Validation"
MODEL_NAME      = "AbujaRain"
PARAMS_PATH     = os.path.join(os.path.dirname(__file__), 'rainfall_best_params.json')

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)

# Initializing the Database
def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{urllib.parse.quote_plus(os.getenv('DB_PASS'))}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

# Feature engineering
def load_and_engineer():
    """Load raw data from DB and build all features."""
    engine = get_db_engine()
    df = pd.read_sql(
        "SELECT * FROM daily_weather WHERE city = city ORDER BY date",
        engine, params={'city': CITY},
        index_col='date', parse_dates=['date']
    )

    # Fix the small number of missing values
    df['pressure']   = df['pressure'].ffill()
    df['visibility'] = df['visibility'].ffill()

    # Create Target — did it rain today?
    df['is_rain'] = (df['precip'] > 0).astype(int)

    # Drop leaky and redundant columns
    to_drop = [
        'precip', 'precipprob', 'precipcover', 'preciptype',
        'conditions', 'description', 'icon', 'severerisk',
        'snow', 'snowdepth', 'feelslike', 'feelslikemax', 'feelslikemin',
        'dew', 'uvindex', 'temp_max', 'temp_min', 'solarenergy',
        'city', 'source', 'stations', 'sunrise', 'sunset', 'windgust'
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    # Lags — how many days back each feature is meaningful (from PACF analysis)
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

    # Deltas — direction of change (lag1 - lag2)
    for col in ['humidity', 'cloudcover', 'visibility']:
        df[f'{col}_delta'] = df[f'{col}_lag1'] - df[f'{col}_lag2']
    df['pressure_delta'] = df['pressure'] - df['pressure_lag1']
    df['temp_avg_delta'] = df['temp_avg'] - df['temp_avg_lag1']

    # Cyclical encoding — month and wind direction are circular
    df['month_sin']   = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos']   = np.cos(2 * np.pi * df.index.month / 12)
    df['winddir_sin'] = np.sin(2 * np.pi * df['winddir'] / 360)
    df['winddir_cos'] = np.cos(2 * np.pi * df['winddir'] / 360)
    df = df.drop(columns=['winddir'])

    # Rolling means — medium term atmospheric state
    for col in ['pressure', 'humidity']:
        df[f'{col}_roll3'] = df[col].shift(1).rolling(3).mean()
        df[f'{col}_roll7'] = df[col].shift(1).rolling(7).mean()

    # Drop raw T0 features — we cant use today to predict today
    t0_to_drop = ['temp_avg', 'humidity', 'pressure', 'cloudcover',
                  'visibility', 'windspeed', 'solarradiation', 'moonphase']
    df = df.drop(columns=[c for c in t0_to_drop if c in df.columns])

    return df.dropna()

# Hyperparameter Tuning
def tune_params(df):
    """Find best hyperparameters using time-series aware cross validation."""
    # Only train on data up to 2024 — keep 2025 unseen for walk-forward
    train = df[df.index.year <= 2024]
    X_tr  = train.drop(columns=['is_rain'])
    y_tr  = train['is_rain']

    param_grid = {
        'n_estimators':  [100, 200, 300],
        'max_depth':     [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample':     [0.7, 0.9, 1.0],
    }

    # TimeSeriesSplit ensures no future data leaks into training during tuning
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1
    )
    grid.fit(X_tr, y_tr)

    print(f"Best CV F1:  {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")
    return grid.best_params_, round(grid.best_score_, 4)

# Perform walk forward Validation (Expanding Window)
def walk_forward_validation(params, df):
    """
    Train on all years before test_year, test on test_year.
    Rolls forward year by year — no future data ever leaks into training.
    """
    scores = []

    for test_year in [2022, 2023, 2024, 2025]:
        train = df[df.index.year < test_year]
        test  = df[df.index.year == test_year]

        if len(test) == 0:
            continue

        X_tr, y_tr = train.drop(columns=['is_rain']), train['is_rain']
        X_te, y_te = test.drop(columns=['is_rain']),  test['is_rain']

        m = XGBClassifier(**params, random_state=42, eval_metric='logloss')
        m.fit(X_tr, y_tr)

        score = f1_score(y_te, m.predict(X_te), pos_label=1)
        scores.append(score)
        print(f"  {test_year} → F1 Rain: {score:.3f}")

    avg_f1 = round(sum(scores) / len(scores), 4)
    print(f"  Average F1: {avg_f1}")
    return avg_f1

# Perform on the test set and save the confusion matrix
def save_confusion_matrix(params, df):
    """Train on 2020-2024, test on 2025, save confusion matrix image."""
    train = df[df.index.year <= 2024]
    test  = df[df.index.year == 2025]

    X_tr, y_tr = train.drop(columns=['is_rain']), train['is_rain']
    X_te, y_te = test.drop(columns=['is_rain']),  test['is_rain']

    m = XGBClassifier(**params, random_state=42, eval_metric='logloss')
    m.fit(X_tr, y_tr)
    y_pred = m.predict(X_te)

    # Confusion matrix metrics
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Save image
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=['No Rain', 'Rain']).plot(cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix — Validation (2025)')
    plt.tight_layout()
    img_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
    plt.savefig(img_path)
    plt.close()

    return tn, fp, fn, tp, img_path

# Main Function
def validate():
    print("Loading and Fixing the data...")
    df = load_and_engineer()
    print(f"Shape: {df.shape}")

    with mlflow.start_run(run_name="Rainfall_Validation"):

        # 1. Tune new params
        print("\nTuning new parameters...")
        new_params, cv_f1 = tune_params(df)

        # 2. Walk-forward validation
        print("\nRunning walk-forward validation...")
        new_avg_f1 = walk_forward_validation(new_params, df)

        # 3. Confusion matrix
        print("\nGenerating confusion matrix...")
        tn, fp, fn, tp, img_path = save_confusion_matrix(new_params, df)

        # 4. Log everything to MLflow
        mlflow.log_params(new_params)
        mlflow.log_metric("cv_f1",       cv_f1)
        mlflow.log_metric("avg_wf_f1",   new_avg_f1)
        mlflow.log_metric("true_positives",  tp)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("true_negatives",  tn)
        mlflow.log_metric("false_negatives", fn)
        mlflow.log_artifact(img_path)

        # 5. Compare against saved params
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH) as f:
                saved = json.load(f)
            saved_f1 = saved['avg_f1']

            print(f"\nSaved params avg F1: {saved_f1}")
            print(f"New params avg F1:   {new_avg_f1}")

            if new_avg_f1 > saved_f1:
                print("New params are better. Saving.")
                json.dump({'params': new_params, 'avg_f1': new_avg_f1}, open(PARAMS_PATH, 'w'))
                mlflow.log_param("validation_result", "improved")
            else:
                print("Saved params are still better. No update.")
                mlflow.log_param("validation_result", "no_change")
        else:
            # First time running — save whatever we found
            print("\nNo saved params found. Saving as baseline.")
            json.dump({'params': new_params, 'avg_f1': new_avg_f1}, open(PARAMS_PATH, 'w'))
            mlflow.log_param("validation_result", "first_run")

        print("\nDone. Run rainfall/train.py to build the production model.")

if __name__ == "__main__":
    validate()