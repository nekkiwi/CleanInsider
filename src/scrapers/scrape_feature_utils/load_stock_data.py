import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration for your Stooq Database ---
# This map translates Stooq's default column names to the keys we are using.
STOOQ_COLUMN_MAP = {
    '<DATE>': 'Date',
    '<OPEN>': 'Open',
    '<HIGH>': 'High',
    '<LOW>': 'Low',
    '<CLOSE>': 'Close',
    '<VOL>': 'Volume'
}

def find_and_load_stock_data(db_path, ticker):
    """
    Recursively searches for a ticker's data file within the database path,
    loads it, and prepares it for processing.
    """
    search_pattern = f"*{ticker.lower()}*.*"
    found_files = list(db_path.rglob(search_pattern))

    if not found_files:
        print(f"   ❌ File for ticker '{ticker}' not found in database.")
        return None

    filepath = found_files[0]
    if len(found_files) > 1:
        print(f"   ⚠️  Multiple files found for '{ticker}'; using '{filepath.name}'")

    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns=STOOQ_COLUMN_MAP)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df = df.set_index('Date')
        return df
    except Exception as e:
        print(f"   ❌ Error reading or processing file '{filepath.name}': {e}")
        return None

def create_timeline_row(ticker, start_date, end_date, key, db_path):
    """
    For a single ticker, fetches local stock data for a specific key
    and creates a dictionary representing a row, with columns for business days only.
    (This function remains unchanged)
    """
    print(f"▶️  Processing {ticker} | {start_date} to {end_date} | Key: '{key}' (Business Days Only)")
    
    row_data = {"Ticker": ticker, "Start Date": start_date, "End Date": end_date}
    
    stock_data = find_and_load_stock_data(db_path, ticker)
    
    key_data = pd.Series(dtype=float)
    if stock_data is not None and key in stock_data.columns:
        key_data = stock_data[key]
    else:
        print(f"   ⚠️  Key '{key}' not available for {ticker}.")

    business_day_range = pd.date_range(start=start_date, end=end_date, freq='B')
    aligned_data = key_data.reindex(business_day_range)

    for i, value in enumerate(aligned_data):
        column_name = f"day_{i}"
        row_data[column_name] = value if pd.notna(value) else np.nan
        
    return row_data

# --- MODIFIED FUNCTION ---
def build_stock_data_df(ticker, start_date, end_date, key, db_path_str):
    """
    Builds a summary DataFrame for a single ticker and date range
    using data from a local database.

    Args:
        ticker (str): The ticker to process.
        start_date (str): The start of the date range.
        end_date (str): The end of the date range.
        key (str): The stock data key to fetch (e.g., "Close").
        db_path_str (str): The path to the root of the local database.

    Returns:
        pd.DataFrame: A single-row DataFrame with the daily stock data.
    """
    db_path = Path(db_path_str)
    if not db_path.is_dir():
        print(f"🚨 DATABASE PATH NOT FOUND: '{db_path_str}'")
        return pd.DataFrame()

    print(f"🛠️  Building daily summary for {ticker} | Key: '{key}'...\n")
    
    # No loop needed, we call the worker function directly
    timeline_dict = create_timeline_row(
        ticker=ticker, 
        start_date=start_date, 
        end_date=end_date,
        key=key,
        db_path=db_path
    )
        
    print("\n✅  Completed summary assembly")
    
    # Convert the single dictionary into a list containing that dictionary
    # so that pd.DataFrame creates a single row.
    df = pd.DataFrame([timeline_dict])
    return df

# --- MODIFIED MAIN BLOCK ---
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # >> 1. CONFIGURE YOUR DATABASE PATH HERE <<
    DB_PATH = "../../data/stooq_database" # <-- IMPORTANT: CHANGE THIS

    # >> 2. DEFINE THE SINGLE INPUTS FOR YOUR ANALYSIS <<
    TICKER_TO_FETCH = "AAPL"
    START_DATE = "2024-01-02"
    END_DATE = "2024-01-10"
    
    # >> 3. DEFINE WHICH STOCK VALUE YOU WANT TO SEE HERE <<
    KEY_TO_FETCH = "Volume" # Options: "Open", "High", "Low", "Close", "Volume"
    # ------------------------------------------------------------------

    print(f"Input: Ticker={TICKER_TO_FETCH}, Start={START_DATE}, End={END_DATE}")
    print("-" * 40)

    # Call the modified function with the single inputs
    stock_data_df = build_stock_data_df(
        TICKER_TO_FETCH, 
        START_DATE, 
        END_DATE, 
        KEY_TO_FETCH, 
        DB_PATH
    )
    
    if not stock_data_df.empty:
        print("\nFinal daily summary:")
        # Round the data for cleaner display
        day_cols = [col for col in stock_data_df.columns if col.startswith('day_')]
        stock_data_df[day_cols] = stock_data_df[day_cols].round(2)
        print(stock_data_df)
