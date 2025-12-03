# file: src/scrapers/data_loader.py

import pandas as pd
from pathlib import Path
import yfinance as yf
from functools import lru_cache

FINAL_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

def read_csv_safe(filepath):
    """Safely read CSV files, handling empty/corrupted files gracefully."""
    try:
        if Path(filepath).stat().st_size == 0:
            # print(f"  [READ-WARN] Empty file detected: {filepath.name}")
            return pd.DataFrame()
        df = pd.read_csv(filepath)
        # print(f"  [READ-SUCCESS] Loaded file {filepath.name} with shape {df.shape}")
        # print(f"  [READ-DEBUG] Columns in {filepath.name}: {df.columns.tolist()}")
        # if not df.empty:
            # print(f"  [READ-DEBUG] Sample data from {filepath.name}:")
            # print(f"    {df.head(2).to_string()}")
        return df
    except pd.errors.EmptyDataError:
        # print(f"  [READ-ERROR] EmptyDataError reading file: {filepath.name}")
        return pd.DataFrame()
    except Exception as e:
        # print(f"  [READ-ERROR] Error reading file {filepath.name}: {e}")
        return pd.DataFrame()

def _standardize_and_clean(df: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    """
    A single, robust function to clean any OHLCV dataframe. This is the
    definitive version designed to handle all known edge cases.
    """
    # print(f"  [CLEAN-START] Processing ticker {ticker} from {source}")
    # print(f"  [CLEAN-DEBUG] Input data shape: {df.shape}")
    
    if df.empty:
        # print(f"  [CLEAN-WARN] Empty DataFrame received for {ticker} from {source}")
        return pd.DataFrame()

    # print(f"  [CLEAN-DEBUG] Input columns: {df.columns.tolist()}")
    
    df_clean = df.copy()

    # --- Step 1: Handle Column Structure ---
    # First, handle yfinance's MultiIndex columns if they exist.
    if isinstance(df_clean.columns, pd.MultiIndex):
        # print(f"  [CLEAN-DEBUG] Detected MultiIndex columns for {ticker}")
        df_clean.columns = df_clean.columns.get_level_values(0)
        # print(f"  [CLEAN-DEBUG] Flattened columns: {df_clean.columns.tolist()}")
    
    # Aggressively standardize all column names to simple, flat strings.
    df_clean.columns = [str(col).lower().replace('<', '').replace('>', '').strip() for col in df_clean.columns]
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep='first')]

    # --- Step 2: Unify Date into the Index (THE CRITICAL FIX) ---
    # If 'date' exists as a column (from local files), set it as the index.
    if 'date' in df_clean.columns:
        # print(f"  [CLEAN-DEBUG] Found 'date' column for {ticker}, converting to datetime")
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        date_na_count = df_clean['date'].isna().sum()
        # print(f"  [CLEAN-DEBUG] {date_na_count} rows had invalid dates and will be dropped")
        # Remove rows where date parsing failed before setting index
        df_clean = df_clean[~df_clean['date'].isna()]
        if not df_clean.empty:
            df_clean.set_index('date', inplace=True)
            # print(f"  [CLEAN-DEBUG] Set date as index. Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    # If the index isn't already a DatetimeIndex (from yfinance), convert it.
    elif not isinstance(df_clean.index, pd.DatetimeIndex):
        # print(f"  [CLEAN-DEBUG] Converting existing index to DatetimeIndex for {ticker}")
        df_clean.index = pd.to_datetime(df_clean.index, errors='coerce')
        # Drop any rows whose index could not be parsed as a date
        index_na_count = df_clean.index.isna().sum()
        # print(f"  [CLEAN-DEBUG] {index_na_count} rows had invalid index dates and will be dropped")
        df_clean = df_clean[~df_clean.index.isna()]
        # if not df_clean.empty:
            # print(f"  [CLEAN-DEBUG] Date range after index conversion: {df_clean.index.min()} to {df_clean.index.max()}")

    # Now that the index is the date, standardize its name
    df_clean.index.name = 'Date'
    
    if df_clean.empty:
        # print(f"  [CLEAN-WARN] DataFrame became empty after date processing for {ticker}")
        return pd.DataFrame()

    # print(f"  [CLEAN-DEBUG] Shape after date processing: {df_clean.shape}")

    # --- Step 3: Standardize OHLCV Data ---
    rename_map = {
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'vol': 'Volume', 
        'volume': 'Volume',
        'adj close': 'Adj Close'  # Handle adjusted close if present
    }
    df_clean.rename(columns=rename_map, inplace=True)
    # print(f"  [CLEAN-DEBUG] Columns after OHLCV rename: {df_clean.columns.tolist()}")
    
    # Check if we have the required columns - BE MORE LENIENT
    existing_cols = [col for col in FINAL_COLS if col in df_clean.columns]
    # print(f"  [CLEAN-DEBUG] Available OHLCV columns for {ticker}: {existing_cols}")
    
    if len(existing_cols) < 4:  # Allow missing volume if necessary
        # print(f"  [CLEAN-ERROR] Not enough OHLCV columns for {ticker} from {source}. Need at least 4, have {len(existing_cols)}")
        return pd.DataFrame()
        
    df_final = df_clean[existing_cols].copy()
    
    # Convert all data to numeric, coercing errors to NaN
    for col in existing_cols:
        before_conversion = df_final[col].notna().sum()
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        after_conversion = df_final[col].notna().sum()
        # if before_conversion != after_conversion:
            # print(f"  [CLEAN-DEBUG] Column {col}: {before_conversion - after_conversion} values became NaN during conversion")
    
    # Only require Close to be valid, be more lenient with other columns
    close_na_count = df_final['Close'].isna().sum()
    # print(f"  [CLEAN-DEBUG] {close_na_count} rows have NaN Close values and will be dropped")
    df_final = df_final.dropna(subset=['Close'])
    
    # --- Step 4: Detect and Apply Split Adjustments (THE DEFINITIVE FIX) ---
    # This ensures that data from any source is properly adjusted.
    if 'Adj Close' not in df_final.columns:
        close_to_prev_close_ratio = df_final['Close'] / df_final['Close'].shift(1)
        # Detect splits (e.g., a 50% price drop is a 2-for-1 split, ratio ~0.5)
        # We look for large drops, typical of 2:1, 3:1, or 4:1 splits.
        split_candidates = close_to_prev_close_ratio[
            (close_to_prev_close_ratio > 0.1) & (close_to_prev_close_ratio < 0.7)
        ]

        for date, ratio in split_candidates.items():
            # Round to the nearest common split ratio (e.g., 0.5, 0.33, 0.25)
            split_ratio = 1 / round(1 / ratio)
            
            # Adjust all prices and volume before this date
            price_cols = ['Open', 'High', 'Low', 'Close']
            df_final.loc[df_final.index < date, price_cols] *= split_ratio
            if 'Volume' in df_final.columns:
                df_final.loc[df_final.index < date, 'Volume'] = (df_final.loc[df_final.index < date, 'Volume'] / split_ratio).round().astype('int64')
    
    # Fill missing volume with 0 if Volume column exists but has NaN values
    if 'Volume' in df_final.columns:
        volume_na_count = df_final['Volume'].isna().sum()
        if volume_na_count > 0:
            df_final['Volume'] = df_final['Volume'].fillna(0)

    if df_final.empty:
        # print(f"  [CLEAN-ERROR] Final DataFrame is empty after all cleaning for {ticker} from {source}")
        return pd.DataFrame()

    # print(f"  [CLEAN-SUCCESS] Final data for {ticker} from {source}:")
    # print(f"    Shape: {df_final.shape}")
    # print(f"    Date range: {df_final.index.min().date()} to {df_final.index.max().date()}")
    # print(f"    Sample data:")
    # print(f"    {df_final.head(2).to_string()}")
    
    return df_final.sort_index()

@lru_cache(maxsize=None)
def load_ohlcv_with_fallback(ticker: str, db_path_str: str, required_start_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Robustly loads OHLCV data by trying yfinance first, then falling back to local files.
    """
    # print(f"\n[LOADER-START] Loading OHLCV for ticker: {ticker}")
    # print(f"[LOADER-INFO] Required start date: {required_start_date.date() if required_start_date else 'None specified'}")
    # print(f"[LOADER-INFO] Database path: {db_path_str}")
    
    # --- Step 1: Try yfinance as the primary, preferred source ---
    # Calculate start date with some buffer
    start_date = required_start_date
    if start_date is not None:
        start_date = start_date - pd.Timedelta(days=30)  # Add some buffer
        # print(f"[LOADER-INFO] Using buffered start date for yfinance: {start_date.date()}")
    
    # try:
    # print(f"[LOADER-ATTEMPT] Trying yfinance for {ticker}...")
    data = yf.download(ticker, start=start_date, progress=False, timeout=15, auto_adjust=True)
    if not data.empty:
        # print(f"[LOADER-SUCCESS] yfinance returned data for {ticker} with shape {data.shape}")
        # print(f"[LOADER-DEBUG] yfinance columns: {data.columns.tolist()}")
        cleaned_df = _standardize_and_clean(data, ticker, source="yfinance")
        if not cleaned_df.empty and len(cleaned_df) > 10:  # Ensure we have reasonable amount of data
            # print(f"[LOADER-FINAL] Using yfinance data for {ticker} - {len(cleaned_df)} rows")
            return cleaned_df
        # else:
            # print(f"[LOADER-WARN] yfinance data for {ticker} failed cleaning or too short ({len(cleaned_df)} rows)")
    # else:
        # print(f"[LOADER-WARN] yfinance returned empty DataFrame for {ticker}")
    # except Exception as e:
        # print(f"[LOADER-ERROR] yfinance download failed for {ticker}: {e}")
        
    # --- Step 2: Fallback to local Stooq database ---
    # print(f"[LOADER-ATTEMPT] Trying local Stooq fallback for {ticker}...")
    # try:
    db_path = Path(db_path_str)
    if not db_path.exists():
        # print(f"[LOADER-ERROR] Database path does not exist: {db_path_str}")
        return pd.DataFrame()
        
    ticker_lower = ticker.lower()
    search_pattern = f"*{ticker_lower}.*txt"
    # print(f"[LOADER-DEBUG] Searching for files matching: {search_pattern}")
    candidate_files = list(db_path.rglob(search_pattern))
    # print(f"[LOADER-DEBUG] Found {len(candidate_files)} candidate files for {ticker}")

    if candidate_files:
        # Prefer exact match, otherwise use first candidate
        exact_match_file = next((f for f in candidate_files if f.stem.lower() == ticker_lower), candidate_files[0])
        # print(f"[LOADER-INFO] Selected file for {ticker}: {exact_match_file}")
        
        # Use safe CSV reading
        local_data = read_csv_safe(exact_match_file)
        if not local_data.empty:
            cleaned_local = _standardize_and_clean(local_data, ticker, source="local_file")
            if not cleaned_local.empty:
                # print(f"[LOADER-FINAL] Using local data for {ticker} - {len(cleaned_local)} rows")
                return cleaned_local
                # else:
                    # print(f"[LOADER-WARN] Local data for {ticker} failed cleaning")
            # else:
                # print(f"[LOADER-WARN] Local file for {ticker} was empty or unreadable")
        # else:
            # print(f"[LOADER-WARN] No local files found for {ticker} matching pattern {search_pattern}")
    # except Exception as e:
        # print(f"[LOADER-ERROR] Local file processing failed for {ticker}: {e}")

    # print(f"[LOADER-FAIL] No OHLCV data found for {ticker} from any source")
    return pd.DataFrame()
