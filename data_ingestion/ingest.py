import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv

# Load environment variables
# Force dotenv to override existing variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(override=True)

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


def update_csv(df):
    """Appends new rows to the CSV, or creates it if it doesn't exist."""
    csv_path = "./we_csv_files/abuja_gapp_fill.csv"
    
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        updated = pd.concat([existing, df], ignore_index=True)
        updated = updated.drop_duplicates(subset=['date', 'city'], keep='last')
        updated.to_csv(csv_path, index=False)
        print(f"CSV updated: {csv_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"CSV created: {csv_path}")

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

    # 3. Process Data
    new_rows = []
    for day in data['days']:
        row = {
            'date':             day.get('datetime'),
            'city':             CITY,
            'source':           'visual_crossing_api',

            # Temperature
            'temp_avg':         day.get('temp'),
            'temp_max':         day.get('tempmax'),
            'temp_min':         day.get('tempmin'),
            'feelslike':        day.get('feelslike'),
            'feelslikemax':     day.get('feelslikemax'),
            'feelslikemin':     day.get('feelslikemin'),

            # Moisture
            'dew':              day.get('dew'),
            'humidity':         day.get('humidity'),
            'precip':           day.get('precip') or 0.0,
            'precipprob':       day.get('precipprob'),
            'precipcover':      day.get('precipcover'),
            'preciptype':       str(day.get('preciptype', [])),
            'snow':             day.get('snow'),
            'snowdepth':        day.get('snowdepth'),

            # Wind
            'windgust':         day.get('windgust'),
            'windspeed':        day.get('windspeed'),
            'winddir':          day.get('winddir'),

            # Atmosphere
            'pressure':         day.get('sealevelpressure'),
            'cloudcover':       day.get('cloudcover'),
            'visibility':       day.get('visibility'),
            'solarradiation':   day.get('solarradiation'),
            'solarenergy':      day.get('solarenergy'),
            'uvindex':          day.get('uvindex'),
            'severerisk':       day.get('severerisk'),
            'moonphase':        day.get('moonphase'),

            # Descriptive
            'conditions':       day.get('conditions'),
            'description':      day.get('description'),
            'icon':             day.get('icon'),
            'sunrise':          day.get('sunrise'),
            'sunset':           day.get('sunset'),
            'stations':         str(day.get('stations', [])),
        }
        new_rows.append(row)

    # 4. Save to DB
    if new_rows:
        df = pd.DataFrame(new_rows)
        print(f"Saving {len(df)} new daily records...")
        
        # Ensure 'date' column is actual date type for SQL
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Upsert so re-running never causes duplicate key errors
        metadata = MetaData()
        table = Table(DB_TABLE, metadata, autoload_with=engine)

        with engine.begin() as conn:
            records = df.where(pd.notnull(df), None).to_dict(orient='records')
            stmt = insert(table).values(records)
            
            # Define update logic: update everything except keys
            update_dict = {
                col.name: col
                for col in stmt.excluded
                if col.name not in ['date', 'city']
            }
            
            conn.execute(stmt.on_conflict_do_update(
                index_elements=['date', 'city'],
                set_=update_dict
            ))
        print("Success.")
        update_csv(df)
    else:
        print("No rows to save.")

if __name__ == "__main__":
    main()