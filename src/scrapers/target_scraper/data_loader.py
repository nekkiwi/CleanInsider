# file: src/scrapers/target_scraper/data_loader.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache

@lru_cache(maxsize=1)
def load_all_ohlcv_data(tickers: tuple, db_path_str: str) -> dict:
    """Loads all necessary OHLCV data into a dictionary for fast lookups."""
    print("--- Loading all required OHLCV data into memory ---")
    ohlcv_data = {}
    db_path = Path(db_path_str)
    
    for ticker in tqdm(tickers, desc="Loading price data"):
        search_pattern = f"*{ticker.lower()}*.*"
        found_files = list(db_path.rglob(search_pattern))
        if not found_files:
            continue
            
        try:
            df = pd.read_csv(found_files[0])
            df.columns = [col.strip('<>').capitalize() for col in df.columns]
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
            df.set_index('Date', inplace=True)
            df = df[df.index.notna()]
            if 'Ticker' in df.columns:
                df.drop(columns=['Ticker'], inplace=True)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Vol']
            if not all(col in df.columns for col in required_cols):
                continue
            
            # Forward-fill and back-fill to handle weekends/holidays for lookups
            ohlcv_data[ticker] = df[required_cols].ffill().bfill()
        except Exception as e:
            print(f"Warning: Could not load {ticker}. Error: {e}")
            continue
            
    return ohlcv_data
