import os
import sys
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv

# Force dotenv to override existing variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(override=True)

# --- CONFIGURATION ---
CSV_FILENAME = "abuja_gapp_fill.csv"
CSV_FOLDER = "./we_csv_files"
TABLE_NAME = "daily_weather"
BATCH_SIZE = 200  # Process 200 rows at a time to prevent "Too Many Parameters" error

def get_db_engine():
    """
    Creates a database engine with safe password handling.
    """
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT', '5432')
    dbname = os.getenv('DB_NAME')

    if not all([user, password, host, dbname]):
        print("ERROR: Missing database credentials in .env")
        sys.exit(1)

    # Encode password to handle special characters safely
    encoded_pass = urllib.parse.quote_plus(password)
    
    return create_engine(f"postgresql://{user}:{encoded_pass}@{host}:{port}/{dbname}")

def upsert_in_chunks(df, engine, chunk_size):
    """
    Inserts data in small batches (chunks) to avoid crashing the DB driver.
    """
    metadata = MetaData()
    table = Table(TABLE_NAME, metadata, autoload_with=engine)
    
    total_rows = len(df)
    print(f"Starting Upsert for {total_rows} rows (Batch size: {chunk_size})...")

    # Connect once, then loop through chunks
    with engine.begin() as conn:
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = start_idx + chunk_size
            chunk = df.iloc[start_idx : end_idx]
            
            # Convert chunk to list of dicts (handle NaNs as NULL)
            records = chunk.where(pd.notnull(chunk), None).to_dict(orient='records')

            # 1. Create Insert Statement
            stmt = insert(table).values(records)

            # 2. Define Update Logic (Update all cols EXCEPT Primary Keys)
            update_dict = {
                col.name: col 
                for col in stmt.excluded 
                if col.name not in ['date', 'city'] 
            }

            # 3. Create Upsert Statement (On Conflict Do Update)
            on_conflict_stmt = stmt.on_conflict_do_update(
                index_elements=['date', 'city'],
                set_=update_dict
            )

            # 4. Execute
            conn.execute(on_conflict_stmt)
            print(f"Processed rows {start_idx} to {min(end_idx, total_rows)}")

    print("SUCCESS: All batches processed.")

def verify_data(engine):
    """
    Prints a quick summary of what is currently in the DB.
    """
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
        count = result.scalar()
        print(f"\n[Verification] Total rows in '{TABLE_NAME}': {count}")

def load_history():
    file_path = os.path.join(CSV_FOLDER, CSV_FILENAME)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return

    print(f"Reading {file_path}...")
    
    try:
        # 1. Read ALL columns
        df = pd.read_csv(file_path)

        # 2. Rename and prep the key columns
        df = df.rename(columns={
            'datetime':         'date',
            'sealevelpressure': 'pressure',
            'temp':             'temp_avg',
            'tempmax':          'temp_max',
            'tempmin':          'temp_min',
            'name':             'city'
        })

        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True).dt.date
        df['precip'] = df['precip'].fillna(0.0)
        df['city'] = 'Abuja'
        df['source'] = 'visual_crossing_csv'

        # 3. Deduplicate
        initial_count = len(df)
        df = df.drop_duplicates(subset=['date', 'city'], keep='last')
        final_count = len(df)

        if initial_count > final_count:
            print(f"Cleaned up {initial_count - final_count} duplicate rows.")

        # 4. Upsert and verify
        engine = get_db_engine()
        upsert_in_chunks(df, engine, BATCH_SIZE)
        verify_data(engine)

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")

if __name__ == "__main__":
    load_history()