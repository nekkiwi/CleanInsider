# file: src/scrapers/data_loader.py

import pandas as pd
from pathlib import Path
import yfinance as yf
from functools import lru_cache

@lru_cache(maxsize=1)
def load_ohlcv_with_fallback(ticker: str, db_path_str: str) -> pd.DataFrame:
    """
    Attempts to load OHLCV data from the local Stooq database first.
    If it fails, it falls back to downloading from Yahoo Finance.
    Returns an empty DataFrame if both sources fail.
    """
    # 1. Attempt to load from the local database
    db_path = Path(db_path_str)
    ticker_lower = ticker.lower()
    
    # Precise file matching logic
    search_pattern = f"*{ticker_lower}*.*"
    candidate_files = list(db_path.rglob(search_pattern))
    exact_match_file = None
    if candidate_files:
        for file in candidate_files:
            filename_stem = file.stem.lower()
            if filename_stem == ticker_lower or filename_stem.startswith(f"{ticker_lower}.") or filename_stem.startswith(f"{ticker_lower}_"):
                exact_match_file = file
                break
    
    if exact_match_file:
        try:
            df = pd.read_csv(exact_match_file)
            df.columns = [col.strip('<>').capitalize() for col in df.columns]
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
            df.set_index('Date', inplace=True)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Vol']
            if all(col in df.columns for col in required_cols):
                df.rename(columns={'Vol': 'Volume'}, inplace=True) # Standardize column name
                return df[required_cols].ffill().bfill()
        except Exception:
            pass # Failed to load local file, proceed to fallback

    # 2. If local loading fails, fall back to Yahoo Finance
    # print(f"   -> Ticker '{ticker}' not found locally or failed to load. Trying yfinance...")
    try:
        data = yf.download(ticker, progress=False, timeout=10, auto_adjust=True)
        if not data.empty:
            # print(f"      -> Successfully downloaded '{ticker}' from yfinance.")
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in data.columns for col in required_cols):
                return data[required_cols]
    except Exception as e:
        # print(f"      -> yfinance download failed for '{ticker}': {e}")
        pass

    # 3. If both sources fail, return an empty DataFrame
    return pd.DataFrame()
