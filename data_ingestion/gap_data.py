import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION
CSV_FILENAME = "abuja_gapp_fill.csv"
CSV_FOLDER = "./we_csv_files"

def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

def process_gap_data():
    file_path = os.path.join(CSV_FOLDER, CSV_FILENAME)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        print("Did you move the downloaded CSV into 'we_csv_files'?")
        return

    print(f"Reading {file_path}...")
    
    try:
        # Read the Visual Crossing CSV
        df = pd.read_csv(file_path)
        
        # Select only the columns we need
        # Visual Crossing names: datetime, temp, humidity
        clean_df = df[['datetime', 'temp', 'humidity']].copy()
        
        # Rename to match our Database Schema
        clean_df = clean_df.rename(columns={
            'datetime': 'timestamp',
            'temp': 'temperature',
            # humidity is already named 'humidity'
        })
        
        # Convert timestamp to proper datetime format
        clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'])
        
        # Add missing columns required by our DB
        clean_df['city'] = 'Abuja'
        clean_df['source'] = 'visual_crossing' # So we know where it came from
        clean_df['pm2_5'] = None # This data source doesn't have PM2.5, so leave null
        clean_df['pm10'] = None
        
        print(f"Prepare to load {len(clean_df)} new days into database...")
        
        # Connect and Upload
        engine = get_db_engine()
        
        # append = Add to existing data, do not delete old history
        clean_df.to_sql('sensor_data', engine, if_exists='append', index=False)
        
        print(" Success! The gap has been filled.")
        print("Now re-run 'python train.py' to let the AI see this new data.")
        
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    process_gap_data()