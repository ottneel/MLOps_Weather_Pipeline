import os
import pandas as pd
import mlflow.statsmodels
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Setup MLflow Connection
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))

def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

def get_data_since_last_training(last_known_date):
    """
    Fetches the 'Gap Data': everything recorded since the model was last trained.
    """
    engine = get_db_engine()
    print(f"Fetching fresh data since {last_known_date}...")
    
    query = """
        SELECT date, temp_avg
        FROM daily_weather
        WHERE temp_avg IS NOT NULL 
        AND date > %(last_date)s
        ORDER BY date ASC
    """
    df = pd.read_sql(query, engine, params={'last_date': last_known_date})
    
    if df.empty:
        return pd.Series(dtype=float)
    
    # Convert date column to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Extract the temp_avg column as a Series with DatetimeIndex
    daily_data = df['temp_avg']
    
    return daily_data

def generate_forecast():
    model_name = "AbujaTemps"
    print(f"Loading latest Production model for '{model_name}'...")
    
    try:
        # 1. Load Production Model
        model_uri = f"models:/{model_name}/Production"
        loaded_model = mlflow.statsmodels.load_model(model_uri)
        
        # 2. Identify the Gap
        # Statsmodels remembers the last date it was trained on
        last_training_date = pd.to_datetime(loaded_model.data.dates[-1])
        print(f"Model internal clock stopped at: {last_training_date.date()}")
        
        # 3. Fetch the Gap Data
        new_data = get_data_since_last_training(last_training_date)
        
        # Filter out any data that overlaps with what the model already has.
        # (This should already be handled by the SQL query, but keeping for safety)
        if not new_data.empty:
            new_data = new_data[new_data.index > last_training_date]
        
        # 4. Update the Model State
        if not new_data.empty:
            print(f"Rolling forward... Adding {len(new_data)} days of recent history.")
            # refit=False updates the history state without slow re-training
            loaded_model = loaded_model.append(new_data, refit=False)
            current_last_date = new_data.index[-1]
        else:
            print("No new data found. Model is already up to date.")
            current_last_date = last_training_date

        # 5. Forecast from the NEW "Today"
        steps = 3
        forecast = loaded_model.forecast(steps=steps)
        
        # Calculate dates starting from the end of the NEW data
        forecast_dates = [current_last_date + timedelta(days=i) for i in range(1, steps + 1)]
        
        # 6. Save
        forecast_df = pd.DataFrame({
            'forecast_date': forecast_dates,
            'city': 'Abuja',
            'predicted_temp': forecast.values,
            'model_version': 'production_rolled',
            'created_at': datetime.now()
        })
        
        print("Predictions generated:")
        print(forecast_df[['forecast_date', 'predicted_temp']])
        
        engine = get_db_engine()
        forecast_df.to_sql('daily_forecasts', engine, if_exists='append', index=False)
        print("Success! Forecasts saved.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_forecast()