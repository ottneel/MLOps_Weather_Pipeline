import os
import sys
import urllib.parse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

def get_database_url():
    """Constructs the database URL safely, handling special characters."""
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    dbname = os.getenv('DB_NAME')

    if not all([user, password, dbname]):
        print("ERROR: Missing DB_USER, DB_PASS, or DB_NAME in .env file.")
        sys.exit(1)

    # SAFETY FIX: Encode password to handle special chars like '@', '#'
    encoded_password = urllib.parse.quote_plus(password)
    return f"postgresql://{user}:{encoded_password}@{host}:{port}/{dbname}"

def setup_db():
    print("--- Starting Database Setup ---")
    db_url = get_database_url()
    
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            print("Connected to Database.")

            # --- TABLE 1: DAILY WEATHER ---
            print("Setting up 'daily_weather' table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_weather (
                    date DATE NOT NULL,
                    city VARCHAR(50) NOT NULL,
                    temp_avg FLOAT,
                    temp_max FLOAT,
                    temp_min FLOAT,
                    humidity FLOAT,
                    precip FLOAT,
                    windspeed FLOAT,
                    pressure FLOAT,
                    cloudcover FLOAT,
                    source VARCHAR(50),
                    -- IMPROVEMENT: Composite Key prevents duplicate city entries for same day
                    PRIMARY KEY (date, city)
                );
            """))

            # IMPROVEMENT: Index for fast lookups by Date and City
            # (Crucial when querying "Get me the last 7 days of weather for London")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_weather_date_city 
                ON daily_weather (date DESC, city);
            """))
            print("Created 'daily_weather' with Indexes.")

            # --- TABLE 2: DAILY FORECASTS ---
            print("Setting up 'daily_forecasts' table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_forecasts (
                    id SERIAL PRIMARY KEY,
                    forecast_date DATE NOT NULL,
                    city VARCHAR(50) NOT NULL,
                    predicted_temp FLOAT,
                    model_version VARCHAR(50),
                    -- IMPROVEMENT: Use TIMESTAMPTZ for audit trails
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """))

            # IMPROVEMENT: Index for comparing forecasts vs actuals efficiently
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_forecast_lookup 
                ON daily_forecasts (forecast_date, city);
            """))
            print("Created 'daily_forecasts' with Indexes.")

            conn.commit()
            print("\nSUCCESS: Database tables are ready and optimized!")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_db()