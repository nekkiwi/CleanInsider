# file: src/scrapers/data_loader.py

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yfinance as yf


# Use an unbounded cache for the duration of the process.
# This is safe as each worker process will have its own cache.
@lru_cache(maxsize=None)
def load_ohlcv_with_fallback(
    ticker: str, db_path_str: str, required_start_date: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Attempts to load OHLCV data from the local Stooq database first.
    If the local data is insufficient or missing, it falls back to downloading
    the required historical range from Yahoo Finance.
    Returns an empty DataFrame if both sources fail.
    """
    db_path = Path(db_path_str)
    ticker_lower = ticker.lower()
    local_df = pd.DataFrame()

    # 1. Attempt to load from the local database
    search_pattern = f"*{ticker_lower}*.*"
    candidate_files = list(db_path.rglob(search_pattern))
    exact_match_file = None
    if candidate_files:
        for file in candidate_files:
            if file.stem.lower() == ticker_lower:
                exact_match_file = file
                break

    if exact_match_file:
        try:
            df = pd.read_csv(exact_match_file)
            df.columns = [col.strip("<>").capitalize() for col in df.columns]
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
            df.set_index("Date", inplace=True)
            required_cols = ["Open", "High", "Low", "Close"]
            # Accommodate both 'Vol' and 'Volume'
            if "Vol" in df.columns:
                df.rename(columns={"Vol": "Volume"}, inplace=True)
            required_cols.append("Volume")

            if all(col in df.columns for col in required_cols):
                local_df = df[required_cols].sort_index().ffill().bfill()
        except Exception:
            pass  # Silently fail to fallback to yfinance

    # 2. Decide if yfinance is needed based on data completeness
    use_yfinance = False
    if local_df.empty:
        # print(f" -> Ticker '{ticker}': No local data found.", end="")
        use_yfinance = True
    elif required_start_date and local_df.index.min() > required_start_date:
        # print(f" -> Ticker '{ticker}': Local data incomplete (starts {local_df.index.min().date()}, need {required_start_date.date()}).", end="")
        use_yfinance = True

    if use_yfinance:
        # print(" Fetching from yfinance...")
        try:
            # Explicitly request the full history needed from the required start date
            data = yf.download(
                ticker,
                start=required_start_date,
                progress=False,
                timeout=10,
                auto_adjust=True,
            )
            if not data.empty:
                # print(f" -> Successfully downloaded '{ticker}' from yfinance.")
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                if all(col in data.columns for col in required_cols):
                    return data[required_cols].sort_index()
            else:
                # If yfinance returns nothing, we must rely on what we have locally
                return local_df
        except Exception as e:
            print(f" -> yfinance download failed for '{ticker}': {e}")
            # Fallback to whatever local data exists if yfinance fails
            return local_df

    # 3. If local data was sufficient, return it
    return local_df
