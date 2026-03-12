import pandas as pd
import mlflow
import mlflow.statsmodels
from mlflow.tracking import MlflowClient
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import ast
import warnings

warnings.filterwarnings('ignore')
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(override=True)

# CONFIGURATION
VALIDATION_EXP_NAME = "Abuja_Temp_Validation"
PRODUCTION_EXP_NAME = "Abuja_Temp_Production"
MODEL_NAME = "AbujaTemps"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(PRODUCTION_EXP_NAME)

def get_full_data():
    """Fetches 100% of data (Train + Test) for final build."""
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    query = "SELECT date, temp_avg FROM daily_weather WHERE temp_avg IS NOT NULL ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    #df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df#['date'].resample('D').mean().interpolate(method='time')

def get_latest_params():
    """Finds the latest run from Validate.py and extracts parameters."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(VALIDATION_EXP_NAME)
    
    if not exp:
        raise ValueError("Validate.py has never been run! No validation experiment found.")
        
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found in validation experiment.")
    
    latest_run = runs[0]
    print(f"Fetching params from Validation Run: {latest_run.info.run_id}")
    
    try:
        # Convert string "(1,1,1)" back to tuple (1,1,1)
        order = ast.literal_eval(latest_run.data.params['best_order'])
        seasonal = ast.literal_eval(latest_run.data.params['best_seasonal_order'])
        return order, seasonal
    except KeyError:
        raise ValueError("Latest run is missing 'best_order' params.")

def train_production():
    # 1. Get Params
    order, seasonal = get_latest_params()
    print(f"Building Production Model with: {order} x {seasonal}")
    
    # 2. Get Data
    data = get_full_data()
    
    with mlflow.start_run(run_name="Production_Build"):
        # 3. Fit Final Model
        model = SARIMAX(
            data, order=order, seasonal_order=seasonal,
            enforce_stationarity=False, enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        
        # 4. Log & Register
        mlflow.log_param("order", str(order))
        mlflow.log_param("seasonal_order", str(seasonal))
        mlflow.log_metric("aic", model_fit.aic)
        
        model_info = mlflow.statsmodels.log_model(
            model_fit, 
            artifact_path="model", 
            registered_model_name=MODEL_NAME
        )
        
        print(f"Model registered as version {model_info.registered_model_version}")

        # 5. PROMOTE TO PRODUCTION
        client = MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_info.registered_model_version,
            stage="Production",
            archive_existing_versions=True 
        )
        print(f"Version {model_info.registered_model_version} promoted to PRODUCTION.")

if __name__ == "__main__":
    train_production()