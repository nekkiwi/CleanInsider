from functools import lru_cache
from pathlib import Path

import pandas as pd
import yfinance as yf

# The one and only format we will ever return
FINAL_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _standardize_and_clean(df: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    """
    A single, robust function to clean any OHLCV dataframe. This is the
    definitive version designed to handle all known edge cases.
    """
    if df.empty:
        return pd.DataFrame()

    df_clean = df.copy()

    # --- Step 1: Handle Column Structure ---
    # First, handle yfinance's MultiIndex columns if they exist.
    if isinstance(df_clean.columns, pd.MultiIndex):
        df_clean.columns = df_clean.columns.get_level_values(0)

    # Aggressively standardize all column names to simple, flat strings.
    df_clean.columns = [
        str(col).lower().replace("<", "").replace(">", "").strip()
        for col in df_clean.columns
    ]
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep="first")]

    # --- Step 2: Unify Date into the Index (THE CRITICAL FIX) ---
    # If 'date' exists as a column (from local files), set it as the index.
    if "date" in df_clean.columns:
        df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
        # Remove rows where date parsing failed before setting index
        df_clean = df_clean[~df_clean["date"].isna()]
        if not df_clean.empty:
            df_clean.set_index("date", inplace=True)
    # If the index isn't already a DatetimeIndex (from yfinance), convert it.
    elif not isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean.index = pd.to_datetime(df_clean.index, errors="coerce")
        # Drop any rows whose index could not be parsed as a date
        df_clean = df_clean[~df_clean.index.isna()]

    # Now that the index is the date, standardize its name
    df_clean.index.name = "Date"

    if df_clean.empty:
        return pd.DataFrame()

    # --- Step 3: Standardize OHLCV Data ---
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "vol": "Volume",
        "volume": "Volume",
        "adj close": "Adj Close",  # Handle adjusted close if present
    }
    df_clean.rename(columns=rename_map, inplace=True)

    # Check if we have the required columns
    existing_cols = [col for col in FINAL_COLS if col in df_clean.columns]
    if len(existing_cols) < 4:  # Allow missing volume if necessary
        # print(f"  [CLEANER-ERROR {ticker}] Missing required columns. Have: {existing_cols}")
        return pd.DataFrame()

    df_final = df_clean[existing_cols].copy()

    # Convert all data to numeric, coercing errors to NaN
    for col in existing_cols:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    # Only require Close to be valid, be more lenient with other columns
    df_final = df_final.dropna(subset=["Close"])

    # Fill missing volume with 0 if Volume column exists but has NaN values
    if "Volume" in df_final.columns:
        df_final["Volume"] = df_final["Volume"].fillna(0)

    if df_final.empty:
        # print(f"  [CLEANER-ERROR {ticker}] No valid data rows after cleaning")
        return pd.DataFrame()

    # print(f"  [CLEANER-SUCCESS {ticker}] Cleaned from {source}. Shape: {df_final.shape}. Dates: {df_final.index.min().date()} to {df_final.index.max().date()}")
    return df_final.sort_index()


@lru_cache(maxsize=None)
def load_ohlcv_with_fallback(
    ticker: str, db_path_str: str, required_start_date: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Robustly loads OHLCV data by trying yfinance first, then falling back to local files.
    """
    # --- Step 1: Try yfinance as the primary, preferred source ---
    # Calculate start date with some buffer
    start_date = required_start_date
    if start_date is not None:
        start_date = start_date - pd.Timedelta(days=30)  # Add some buffer

    data = yf.download(
        ticker, start=start_date, progress=False, timeout=15, auto_adjust=False
    )
    if not data.empty:
        cleaned_df = _standardize_and_clean(data, ticker, source="yfinance")
        if (
            not cleaned_df.empty and len(cleaned_df) > 10
        ):  # Ensure we have reasonable amount of data
            return cleaned_df
    else:
        pass
        # print(f"  [LOADER-WARN {ticker}] yfinance returned empty DataFrame")

    # --- Step 2: Fallback to local Stooq database ---
    # print(f"  [LOADER-INFO {ticker}] Trying local Stooq fallback...")
    db_path = Path(db_path_str)
    if not db_path.exists():
        # print(f"  [LOADER-ERROR {ticker}] Database path does not exist: {db_path_str}")
        return pd.DataFrame()

    ticker_lower = ticker.lower()
    search_pattern = f"*{ticker_lower}.*txt"
    candidate_files = list(db_path.rglob(search_pattern))

    if candidate_files:
        # Prefer exact match, otherwise use first candidate
        exact_match_file = next(
            (f for f in candidate_files if f.stem.lower() == ticker_lower),
            candidate_files[0],
        )
        # print(f"  [LOADER-INFO {ticker}] Found local file: {exact_match_file}")

        local_data = pd.read_csv(exact_match_file)
        if not local_data.empty:
            cleaned_local = _standardize_and_clean(
                local_data, ticker, source="local_file"
            )
            if not cleaned_local.empty:
                return cleaned_local
    else:
        pass
        # print(f"  [LOADER-WARN {ticker}] No local files found matching pattern: {search_pattern}")

    # print(f"[LOADER-FAIL {ticker}] All sources failed. Returning empty DataFrame.")
    return pd.DataFrame()
