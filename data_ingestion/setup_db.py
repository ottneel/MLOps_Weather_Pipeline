import os
import sys
import urllib.parse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Helps us to Load environment variables from the .env file to the environment.
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

    # Encode password to handle special chars like '@', '#' So it doesn't break if there are special characters in the password.
    encoded_password = urllib.parse.quote_plus(password)
    return f"postgresql://{user}:{encoded_password}@{host}:{port}/{dbname}"

def setup_db():
    print("Starting Database Setup...")
    # we get the db url using the get_database_url function.
    db_url = get_database_url()

    # try and except block to prepare, connect to the database and Create the Necessary Tables.
    # exit the script safely should an error happen.
    try:
        # prepare to connect to the database
        engine = create_engine(db_url)
        # connect to the database safely using a with clause for Resource Management.
        with engine.connect() as conn:
            print("Connected to Database.")

            # Creating the daily_weather table
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
                    -- Composite key using date and city
                    PRIMARY KEY (date, city)
                );
            """))

            # Index for faster lookups by Date and City
            # (Crucial when querying "Get me the last 7 days of weather for Abuja")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_weather_date_city 
                ON daily_weather (date DESC, city);
            """))
            print("Created 'daily_weather' with Indexes.")

            # Create Daily Forecast table to hold the predictions.
            conn.execute(text("DROP TABLE IF EXISTS daily_forecasts CASCADE;"))
            print("Setting up 'daily_forecasts' table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_forecasts (
                    id SERIAL PRIMARY KEY,
                    forecast_date DATE NOT NULL,
                    city VARCHAR(50) NOT NULL,
                    predicted_temp FLOAT,
                    model_version VARCHAR(50),
                    -- Use TIMESTAMPTZ for audit trails for when I move storage to the cloud
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """))

            # Index for comparing forecasts vs actuals efficiently
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_forecast_lookup 
                ON daily_forecasts (forecast_date, city);
            """))
            print("Created 'daily_forecasts' with Indexes.")

            #Saves the Changes Made to the dB
            conn.commit()
            print("\nSUCCESS: Database tables are ready and optimized!")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_db()
