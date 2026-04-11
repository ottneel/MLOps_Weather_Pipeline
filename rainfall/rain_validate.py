import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from rain_features_eng import load_and_engineer
from dotenv import load_dotenv


# Initial Setup
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path, override=True)


EXPERIMENT_NAME = "Abuja_Rain_Validation"



mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)


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
        mlflow.log_artifact(img_path)
        os.remove(img_path)  # clean up local file after MLflow has it stored

        # 5. Compare against previous MLflow run
        # Fetch last 2 runs — max_results=2 is the ceiling,
        # but on the very first run only 1 will exist hence the len() check below
        client = MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=2
        )

        if len(runs) >= 2:
            # Second item is the previous run (most recent is current)
            prev_f1 = float(runs[1].data.metrics.get('avg_wf_f1', 0))
            print(f"\nPrevious avg F1: {prev_f1}")
            print(f"New avg F1:      {new_avg_f1}")

            if new_avg_f1 > prev_f1:
                print("New params are better.")
                mlflow.log_param("validation_result", "improved")
            else:
                print("Previous params are still better.")
                mlflow.log_param("validation_result", "no_change")
        else:
            # Only 1 run exists — nothing to compare against yet
            print("First validation run. Nothing to compare against.")
            mlflow.log_param("validation_result", "first_run")

        print("\nDone. Run rainfall/rain_train.py to build the production model.")

if __name__ == "__main__":
    validate()