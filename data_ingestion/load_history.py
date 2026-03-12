import os
import sys
import urllib.parse
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv

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
            print(f"   ✓ Processed rows {start_idx} to {min(end_idx, total_rows)}")

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
        # 1. Read and Prep Data
        df = pd.read_csv(file_path)
        
        required_cols = [
            'datetime', 'temp', 'tempmax', 'tempmin', 
            'humidity', 'precip', 'windspeed', 
            'sealevelpressure', 'cloudcover'
        ]
        
        # Validation
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"ERROR: CSV missing columns: {missing}")
            return

        clean_df = df[required_cols].copy()
        
        clean_df = clean_df.rename(columns={
            'datetime': 'date',
            'temp': 'temp_avg',
            'tempmax': 'temp_max',
            'tempmin': 'temp_min',
            'sealevelpressure': 'pressure'
        })
        
        clean_df['date'] = pd.to_datetime(clean_df['date'], format='mixed',dayfirst=True).dt.date
        clean_df['precip'] = clean_df['precip'].fillna(0.0)
        clean_df['city'] = 'Abuja'
        clean_df['source'] = 'visual_crossing_csv'

        # --- NEW: METHOD 1 (Pandas Deduplication) ---
        # This removes duplicates inside the CSV before they ever reach the DB.
        # We keep 'last' assuming the later row in the file is the most corrected version.
        initial_count = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=['date', 'city'], keep='last')
        final_count = len(clean_df)

        if initial_count > final_count:
            print(f"⚠ Cleaned up {initial_count - final_count} duplicate rows in the CSV itself.")
        # --------------------------------------------

        # 2. Run the Chunked Upsert
        engine = get_db_engine()
        upsert_in_chunks(clean_df, engine, BATCH_SIZE)
        
        # 3. Verify
        verify_data(engine)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")

if __name__ == "__main__":
    load_history()