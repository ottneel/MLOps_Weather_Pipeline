import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
from rain_features_eng import load_and_engineer
from dotenv import load_dotenv

# ── SETUP ─────────────────────────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path, override=True)

EXPERIMENT_NAME = "Abuja_Rain_Production"
MODEL_NAME      = "AbujaRain"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)

# ── FETCH BEST PARAMS FROM MLFLOW ─────────────────────────────────────────────
def get_latest_params():
    """Fetch best params from the latest validate.py run in MLflow."""
    client = MlflowClient()
    exp = client.get_experiment_by_name("Abuja_Rain_Validation")

    if not exp:
        raise ValueError("No validation experiment found. Run validate.py first.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found. Run validate.py first.")

    latest_run = runs[0]
    print(f"Fetching params from validation run: {latest_run.info.run_id}")

    # Cast types explicitly — MLflow stores everything as strings
    params = {
        'n_estimators':  int(latest_run.data.params['n_estimators']),
        'max_depth':     int(latest_run.data.params['max_depth']),
        'learning_rate': float(latest_run.data.params['learning_rate']),
        'subsample':     float(latest_run.data.params['subsample']),
    }
    avg_f1 = float(latest_run.data.metrics['avg_wf_f1'])

    return params, avg_f1

# ── MAIN ──────────────────────────────────────────────────────────────────────
def train():
    # 1. Get best params from MLflow
    params, avg_f1 = get_latest_params()
    print(f"Training with params: {params}")
    print(f"Expected walk-forward F1: {avg_f1}")

    # 2. Load full dataset
    print("\nLoading full dataset...")
    df = load_and_engineer()
    print(f"Training on {len(df)} rows")

    X = df.drop(columns=['is_rain'])
    y = df['is_rain']

    with mlflow.start_run(run_name="Rainfall_Production_Build"):

        # 3. Train on full data
        model = XGBClassifier(**params, random_state=42, eval_metric='logloss')
        model.fit(X, y)

        # 4. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("avg_wf_f1",     avg_f1)
        mlflow.log_metric("training_rows", len(df))

        # 5. Register model in MLflow
        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        print(f"\nModel registered as version {model_info.registered_model_version}")

        # 6. Promote to Production — archives all previous versions automatically
        client = MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_info.registered_model_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Version {model_info.registered_model_version} promoted to Production.")
        print("\nDone. Run rainfall/rain_predict.py to generate today's forecast.")

if __name__ == "__main__":
    train()