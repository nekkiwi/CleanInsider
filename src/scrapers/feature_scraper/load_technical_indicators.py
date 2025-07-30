# file: /src/scrapers/load_technical_indicators.py (Using 'ta' library with Joblib + TQDM)

from functools import lru_cache
from pathlib import Path

import pandas as pd

# The 'ta' library for technical analysis
import ta

# Joblib is the robust choice for parallel data workloads
from joblib import Parallel, delayed
from tqdm import tqdm

# --- DATA LOADING HELPER (Unchanged) ---
STOOQ_COLUMN_MAP = {
    "<DATE>": "Date",
    "<OPEN>": "Open",
    "<HIGH>": "High",
    "<LOW>": "Low",
    "<CLOSE>": "Close",
    "<VOL>": "Volume",
}


@lru_cache(maxsize=512)
def find_and_load_ohlcv_data(db_path_str: str, ticker: str) -> pd.DataFrame:
    # This function is correct and remains unchanged
    db_path = Path(db_path_str)
    search_pattern = f"*{ticker.lower()}*.*"
    found_files = list(db_path.rglob(search_pattern))
    if not found_files:
        return pd.DataFrame()
    filepath = found_files[0]
    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns=STOOQ_COLUMN_MAP)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df = df.set_index("Date")
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[ohlcv_cols].apply(pd.to_numeric, errors="coerce").dropna()
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


# --- REWRITTEN INDICATOR CALCULATION USING 'ta' LIBRARY ---
def calculate_indicators(
    stock_df: pd.DataFrame, is_market_instrument=False
) -> pd.DataFrame:
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="ta.*")

    if stock_df.empty or len(stock_df) < 20:  # Need enough data for most indicators
        return stock_df

    # Create a copy to avoid SettingWithCopyWarning
    df = stock_df.copy()

    # Add all indicators and handle potential errors if data is too short
    try:
        # Volume Indicators
        df = ta.add_volume_ta(df, "High", "Low", "Close", "Volume", fillna=True)
        # Volatility Indicators
        df = ta.add_volatility_ta(df, "High", "Low", "Close", fillna=True)
        # Trend Indicators
        df = ta.add_trend_ta(df, "High", "Low", "Close", fillna=True)
        # Momentum Indicators
        df = ta.add_momentum_ta(df, "High", "Low", "Close", "Volume", fillna=True)
    except Exception:
        # This can happen with very short data series, it's safe to just return what we have
        pass

    # Manually calculate 52-week high/low distance
    rolling_high_52w = df["Close"].rolling(window=252, min_periods=1).max()
    rolling_low_52w = df["Close"].rolling(window=252, min_periods=1).min()
    df["Dist_52w_High_Pct"] = (df["Close"] - rolling_high_52w) / rolling_high_52w
    df["Dist_52w_Low_Pct"] = (df["Close"] - rolling_low_52w) / rolling_low_52w

    # If it's a market instrument, we only keep a subset of columns
    if is_market_instrument:
        market_cols_to_keep = [
            "trend_sma_fast",
            "trend_sma_slow",
            "momentum_rsi",
            "trend_macd",
            "trend_macd_signal",
            "trend_macd_diff",
            "trend_adx",
            "volatility_bbm",
            "volatility_bbh",
            "volatility_bbl",
        ]
        # Keep original OHLCV + Dist + the selected indicators
        final_cols = list(stock_df.columns) + ["Dist_52w_High_Pct", "Dist_52w_Low_Pct"]
        final_cols += [col for col in market_cols_to_keep if col in df.columns]
        df = df[final_cols]

    # Cast any new integer columns to float to prevent type warnings later
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float64")

    return df


# --- PARALLEL WORKER (Unchanged, still robust) ---
def _process_ticker_batch(
    work_item: tuple, db_path_str: str, market_data: dict
) -> list[dict]:
    # This worker function is correct and needs no changes
    import warnings

    warnings.filterwarnings("ignore")

    ticker, dates_to_process = work_item
    stock_df_raw = find_and_load_ohlcv_data(db_path_str, ticker)
    if stock_df_raw.empty:
        return []

    stock_df_indicators = calculate_indicators(stock_df_raw, is_market_instrument=False)

    spx_df, vixy_df = market_data.get("spx"), market_data.get("vixy")
    if spx_df is None or vixy_df is None:
        return []

    ticker_rows = []
    for date_str in dates_to_process:
        target_date = pd.to_datetime(date_str)
        try:
            stock_features = stock_df_indicators.loc[:target_date].iloc[-1]
            spx_features = spx_df.loc[:target_date].iloc[-1]
            vixy_features = vixy_df.loc[:target_date].iloc[-1]
        except IndexError:
            continue

        result_dict = stock_features.to_dict()
        result_dict["Days_Since_IPO"] = (
            target_date - stock_df_indicators.index.min()
        ).days
        result_dict.update(spx_features.add_prefix("Market_SPX_").to_dict())
        result_dict.update(vixy_features.add_prefix("Market_VIXY_").to_dict())
        result_dict["Ticker"], result_dict["Filing Date"] = ticker, date_str
        ticker_rows.append(result_dict)

    return ticker_rows


# --- MAIN ORCHESTRATOR USING JOBLIB (Unchanged, still robust) ---
def load_technical_indicators_df(
    input_df: pd.DataFrame, db_path_str: str
) -> pd.DataFrame:
    print("--- Pre-loading market instruments (SPX, VIXY) ---")
    market_data = {
        "spx": calculate_indicators(
            find_and_load_ohlcv_data(db_path_str, "^spx"), is_market_instrument=True
        ),
        "vixy": calculate_indicators(
            find_and_load_ohlcv_data(db_path_str, "vixy.us"), is_market_instrument=True
        ),
    }

    work_items = list(input_df.groupby("Ticker")["Filing Date"].apply(list).items())
    print(f"--- Starting parallel processing for {len(work_items)} unique tickers ---")
    n_jobs = -2

    tasks = [
        delayed(_process_ticker_batch)(item, db_path_str, market_data)
        for item in work_items
    ]
    results = Parallel(n_jobs=n_jobs)(tqdm(tasks, desc="Processing Tickers"))

    all_rows = [row for ticker_rows in results for row in ticker_rows if ticker_rows]
    if not all_rows:
        return pd.DataFrame()

    final_df = pd.DataFrame(all_rows)

    # Define the raw columns to drop
    raw_ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    # Drop them from the final DataFrame, ignoring errors if they don't exist
    final_df.drop(columns=raw_ohlcv_cols, inplace=True, errors="ignore")
    print("   → Removed raw OHLCV columns from the final technical features set.")

    cols = ["Ticker", "Filing Date"] + [
        col for col in final_df.columns if col not in ["Ticker", "Filing Date"]
    ]
    return final_df[cols]


# --- STANDALONE TEST BLOCK (Unchanged) ---
if __name__ == "__main__":
    print("Running load_technical_indicators.py in standalone test mode...")
    DB_PATH = "../../data/stooq_database"
    data = {
        "Ticker": ["AAPL", "MSFT", "TSLA"],
        "Filing Date": ["2024-01-10", "2023-12-29", "2024-03-15"],
    }
    input_df = pd.DataFrame(data)
    summary_df = load_technical_indicators_df(input_df, DB_PATH)
    if not summary_df.empty:
        print("\nStandalone test complete. Sample output:")
        print(summary_df.head())
        summary_df.to_csv("technical_features_summary_standalone.csv", index=False)
