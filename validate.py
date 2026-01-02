import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels
import pmdarima as pm
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import warnings

# Safety for server environments (prevents "no display name" errors)
plt.switch_backend('Agg') 

warnings.filterwarnings('ignore')
load_dotenv()

# CONFIGURATION
EXPERIMENT_NAME = "Abuja_Temp_Validation"
MODEL_NAME = "AbujaTemps"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)

def get_clean_data():
    """Fetches data from DB and resamples to Daily Mean."""
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    
    query = "SELECT date, temp_avg FROM daily_weather WHERE temp_avg IS NOT NULL ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    #df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df#['date'].resample('D').mean().interpolate(method='time')

def analyze_data_properties(data):
    """
    Runs ADF test, plots Decomposition, ACF, and PACF.
    Logs everything to MLflow.
    """
    print("Running Stationarity & Seasonality Analysis...")

    # 1. Augmented Dickey-Fuller Test (ADF)
    # H0: The time series is non-stationary.
    # H1: The time series is stationary.
    adf_result = adfuller(data.dropna())
    
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    
    print(f"ADF Statistic: {adf_stat}")
    print(f"P-Value: {p_value}")
    
    mlflow.log_metric("adf_p_value", p_value)
    mlflow.log_metric("adf_statistic", adf_stat)
    
    # Interpretation: If p > 0.05, we fail to reject H0 (It IS Non-Stationary)
    is_stationary = p_value < 0.05
    mlflow.log_param("is_data_stationary_raw", str(is_stationary))

    # 2. Seasonality Decomposition Plot
    # We use period = 365
    decomp = seasonal_decompose(data, model='additive', period=365)
    fig_decomp = decomp.plot()
    fig_decomp.set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig("seasonality_decomposition.png")
    mlflow.log_artifact("seasonality_decomposition.png")
    plt.close(fig_decomp)

    # 3. ACF and PACF Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    plot_acf(data, ax=ax1, lags=40)
    ax1.set_title("Autocorrelation Function (ACF)")
    
    plot_pacf(data, ax=ax2, lags=40)
    ax2.set_title("Partial Autocorrelation Function (PACF)")
    
    plt.tight_layout()
    plt.savefig("acf_pacf_plots.png")
    mlflow.log_artifact("acf_pacf_plots.png")
    plt.close(fig)
    
    # Cleanup local files
    if os.path.exists("seasonality_decomposition.png"): os.remove("seasonality_decomposition.png")
    if os.path.exists("acf_pacf_plots.png"): os.remove("acf_pacf_plots.png")

def evaluate_params(train_data, test_data, order, seasonal_order, name):
    """Trains on 'Past', Validates on 'Future' (Test Set). Returns MAE."""
    print(f"Testing {name}: {order} x {seasonal_order}")
    
    # 1. Fit on Train Data
    model = SARIMAX(
        train_data, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # 2. Walk-Forward Validation on Test Data
    predictions = []
    history = model_fit
    
    for t in range(len(test_data)):
        # Predict one step ahead
        yhat = history.forecast(steps=1).iloc[0]
        predictions.append(yhat)
        # Update history with real observation (no full refit)
        history = history.append([test_data.iloc[t]], refit=False)
        
    mae = mean_absolute_error(test_data, predictions)
    print(f"Result: MAE = {mae:.4f}")
    return mae, predictions

def evaluate():
    data = get_clean_data()
    if len(data) < 50: 
        print("Not enough data to run validation.")
        return

    # Split: Train on first 80%, Test on last 20%
    split = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:split], data.iloc[split:]
    
    with mlflow.start_run(run_name="Current_vs_New"):
        
        # --- NEW STEP: Analyze Data Properties before Modeling ---
        # We only look at Train Data to avoid Data Leakage (peeking at the test set)
        analyze_data_properties(train_data)
        # ---------------------------------------------------------

        # 1. Current Production Model
        current_mae = float('inf')
        current_order = None
        current_seasonal = None
        
        try:
            print("Fetching current Production model...")
            # Load specifically the model tagged 'Production' in Registry
            prod_model = mlflow.statsmodels.load_model(f"models:/{MODEL_NAME}/Production")
            current_order = prod_model.model.order
            current_seasonal = prod_model.model.seasonal_order
            
            current_mae, current_preds = evaluate_params(train_data, test_data, current_order, current_seasonal, "Current")
        except Exception:
            print("No Production model found. Current is disqualified.")

        # 2. New Auto-ARIMA Search
        print("Running Auto-ARIMA to find better Parameters...")
        auto = pm.auto_arima(
            train_data, start_p=0, start_q=0, max_p=3, max_q=3, m=7,
            seasonal=True, stepwise=True, suppress_warnings=True, error_action='ignore'
        )
        new_mae, new_preds = evaluate_params(train_data, test_data, auto.order, auto.seasonal_order, "New")

        # 3. THE SELECTION DECISION
        # New Parameters must improve by at least 0.05 MAE to replace Current
        improvement_threshold = 0.05
        
        if current_mae == float('inf'):
            winner = "New (Default)"
            w_order, w_seasonal, w_mae, w_preds = auto.order, auto.seasonal_order, new_mae, new_preds
        elif new_mae < (current_mae - improvement_threshold):
            winner = "New (New Params)"
            w_order, w_seasonal, w_mae, w_preds = auto.order, auto.seasonal_order, new_mae, new_preds
        else:
            winner = "Current (Retain Old)"
            w_order, w_seasonal, w_mae, w_preds = current_order, current_seasonal, current_mae, current_preds

        print(f"\n SELECTED WINNER: {winner} (Overall MAE: {w_mae:.4f})")

        # 4. THE QUALITY CHECK (SAFETY CHECK)

        # Calculate MAE specifically for the last 3 days of the test set
        last_3_actuals = test_data.iloc[-3:]
        last_3_forecasts = w_preds[-3:]
        
        check_mae = mean_absolute_error(last_3_actuals, last_3_forecasts)
        
        print("\n QUALITY CHECK")
        print(f"Last 3 Days Actuals:  {last_3_actuals.values}")
        print(f"Last 3 Days Forecast: {[round(x, 2) for x in last_3_forecasts]}")
        print(f"Recent MAE: {check_mae:.4f} (Threshold: 2.0)")

        if check_mae > 2.0:
            # LOG FAILURE AND CRASH
            mlflow.log_param("quality_check", "FAILED")
            mlflow.log_metric("check_mae", check_mae)
            
            error_msg = f"CRITICAL: Deployment Halted. Recent MAE ({check_mae:.2f}) > 2.0."
            print(f"{error_msg}")
            
            # Raise error to stop CI/CD pipeline with a non-zero exit code
            raise ValueError(error_msg)
        
        print("Quality Check PASSED. Proceeding to log parameters.")
        mlflow.log_param("quality_check", "PASSED")
        mlflow.log_metric("check_mae", check_mae)

        # 5. LOG RESULTS (Only happens if Check Passes)
        mlflow.log_param("winner", winner)
        # CRITICAL: Log these as strings so Training Script can read them
        mlflow.log_param("best_order", str(w_order))
        mlflow.log_param("best_seasonal_order", str(w_seasonal))
        mlflow.log_metric("val_mae", w_mae)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(test_data.index, test_data, label='Actual')
        plt.plot(test_data.index, w_preds, color='red', linestyle='--', label=f'Forecast ({winner})')
        plt.title(f'Validation Winner: {winner}')
        plt.legend()
        plt.savefig("winner.png")
        mlflow.log_artifact("winner.png")
        if os.path.exists("winner.png"): os.remove("winner.png")

if __name__ == "__main__":
    evaluate()
