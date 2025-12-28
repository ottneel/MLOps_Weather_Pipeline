import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
VC_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
CITY = "Abuja"
DB_TABLE = "daily_weather" 

# Visual Crossing Base URL
BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

def get_db_engine():
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

def get_last_recorded_date(engine):
    """Finds the last date recorded in the DB."""
    with engine.connect() as conn:
        try:
            # We look for the latest date stored
            query = text(f"SELECT MAX(date) FROM {DB_TABLE} WHERE city = :city")
            result = conn.execute(query, {"city": CITY}).scalar()
            
            # If result is a datetime, convert to date; if None, return None
            if isinstance(result, datetime):
                return result.date()
            return result
        except Exception:
            # Table might not exist yet
            return None

def fetch_weather_data(start_date, end_date):
    """
    Fetches DAILY summary data for a date range.
    """
    # Visual Crossing URL structure for Daily data
    # We explicitly ask for metric units
    url = f"{BASE_URL}/{CITY}/{start_date}/{end_date}?unitGroup=metric&key={VC_API_KEY}&contentType=json"
    
    print(f"Fetching daily data from {start_date} to {end_date}...")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        return None
    
    return response.json()

def main():
    engine = get_db_engine()
    
    # 1. Determine Date Range
    last_date = get_last_recorded_date(engine)
    
    # We want data up to YESTERDAY. 
    yesterday = (datetime.now() - timedelta(days=1)).date()
    
    if last_date is None:
        print("No data found in DB. Initializing with last 30 days.")
        start_date = yesterday - timedelta(days=30)
    else:
        start_date = last_date + timedelta(days=1)

    if start_date > yesterday:
        print("Database is already up to date (records exist up to yesterday).")
        return

    # 2. Fetch Data
    data = fetch_weather_data(start_date, yesterday)
    
    if not data or 'days' not in data:
        print("No data received from API.")
        return

    # 3. Process Data (Modified to include specific requested fields)
    new_rows = []
    for day in data['days']:
        row = {
            'date': day['datetime'],           # YYYY-MM-DD
            'city': CITY,
            'temp_avg': day.get('temp'),
            'temp_max': day.get('tempmax'),
            'temp_min': day.get('tempmin'),
            'humidity': day.get('humidity'),
            'precip': day.get('precip'),
            'windspeed': day.get('windspeed'),   
            'pressure': day.get('pressure'),     
            'cloudcover': day.get('cloudcover'), 
            'source': day.get('source')          
        }
        new_rows.append(row)

    # 4. Save to DB
    if new_rows:
        df = pd.DataFrame(new_rows)
        print(f"Saving {len(df)} new daily records...")
        
        # Ensure 'date' column is actual date type for SQL
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        df.to_sql(DB_TABLE, engine, if_exists='append', index=False)
        print("Success.")
    else:
        print("No rows to save.")

if __name__ == "__main__":
    main()