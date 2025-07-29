# file: calculate_technical_indicators_fast.py (Fully Cleaned-up)

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
import warnings
import os
import concurrent.futures
from tqdm import tqdm

# Monkey-patch numpy to support pandas_ta expectations
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# --- Helper functions (find_and_load_ohlcv_data, calculate_indicators) are correct and unchanged ---
STOOQ_COLUMN_MAP = {
    '<DATE>': 'Date', '<OPEN>': 'Open', '<HIGH>': 'High',
    '<LOW>': 'Low', '<CLOSE>': 'Close', '<VOL>': 'Volume'
}

@lru_cache(maxsize=256)
def find_and_load_ohlcv_data(db_path_str: str, ticker: str) -> pd.DataFrame:
    # (This function is unchanged)
    db_path = Path(db_path_str)
    search_pattern = f"*{ticker.lower()}*.*"
    found_files = list(db_path.rglob(search_pattern))
    if not found_files: return pd.DataFrame()
    filepath = found_files[0]
    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns=STOOQ_COLUMN_MAP)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df = df.set_index('Date')
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[ohlcv_cols]
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df.sort_index()
    except Exception:
        return pd.DataFrame()

def calculate_indicators(stock_df: pd.DataFrame, is_market_instrument=False) -> pd.DataFrame:
    # Import is correctly placed here for parallel safety
    import pandas_ta as ta
    # (This function is unchanged)
    if stock_df.empty: return pd.DataFrame()
    rolling_high_52w = stock_df['Close'].rolling(window=252, min_periods=1).max()
    rolling_low_52w = stock_df['Close'].rolling(window=252, min_periods=1).min()
    stock_df['Dist_52w_High_Pct'] = (stock_df['Close'] - rolling_high_52w) / rolling_high_52w
    stock_df['Dist_52w_Low_Pct'] = (stock_df['Close'] - rolling_low_52w) / rolling_low_52w
    # (Strategies remain the same)
    if is_market_instrument:
        strategy = ta.Strategy(name="Market Indicators", ta=[
            {"kind": "sma", "length": 50}, {"kind": "sma", "length": 200},
            {"kind": "rsi"}, {"kind": "macd"}, {"kind": "adx"}, {"kind": "bbands"},
        ])
    else: # Full strategy for the primary ticker
        strategy = ta.Strategy(name="All Indicators", ta=[
            {"kind": "rsi"}, {"kind": "macd"}, {"kind": "cci"}, {"kind": "roc"}, {"kind": "ao"},
            {"kind": "bop"}, {"kind": "mom"}, {"kind": "ppo"}, {"kind": "trix"}, {"kind": "tsi"},
            {"kind": "ema", "length": 20}, {"kind": "sma", "length": 20},
            {"kind": "vwap"}, {"kind": "vortex"}, {"kind": "atr"}, {"kind": "donchian"},
            {"kind": "kc"}, {"kind": "obv"}, {"kind": "cmf"}, {"kind": "mfi"}, {"kind": "efi"},
        ])
    stock_df.ta.strategy(strategy)
    for col in stock_df.columns:
        if stock_df[col].dtype == 'int64':
            stock_df[col] = stock_df[col].astype('float64')
    return stock_df

# --- Worker function is correct and unchanged ---
def load_and_process_ticker(ticker: str, db_path_str: str):
    # This is the bulletproof way to suppress warnings in the worker
    warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta.*")
    
    df = find_and_load_ohlcv_data(db_path_str, ticker)
    if df.empty:
        return (ticker, None)
    
    is_market = ticker.lower() in ["^spx", "vixy.us"]
    indicators_df = calculate_indicators(df, is_market_instrument=is_market)
    
    return (ticker, indicators_df)

# --- CHANGE: Pre-computation function signature is now clean ---
def preload_and_calculate_all_data(tickers: list, db_path_str: str) -> dict:
    data_warehouse = {}
    
    # We can define the number of workers here. Using half the CPUs is a good
    # choice to keep the machine responsive. This logic is now self-contained.
    max_workers = max(1, os.cpu_count() // 2) 
    print(f"--- Starting Parallel Technical Indicator Pre-computation on {max_workers} cores ---")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(load_and_process_ticker, tickers, [db_path_str] * len(tickers))
        
        for result in tqdm(results_iterator, total=len(tickers), desc="Pre-computing Indicators"):
            ticker, df = result
            if df is not None:
                data_warehouse[ticker] = df

    print("--- ✅ Pre-computation Phase Complete ---\n")
    return data_warehouse

# --- CHANGE: Main orchestrator signature is now clean ---
def load_technical_indicators_df(input_df: pd.DataFrame, db_path_str: str) -> pd.DataFrame:
    """
    Takes a DataFrame and returns a summary with all technical features.
    Parallelism is managed internally.
    """
    unique_tickers = input_df['Ticker'].unique().tolist()
    market_tickers = ["^spx", "vixy.us"]
    all_needed_tickers = sorted(list(set(unique_tickers + market_tickers)))

    # The call to the pre-computation function is now simpler
    data_warehouse = preload_and_calculate_all_data(all_needed_tickers, db_path_str)

    # The rest of the function remains the same.
    print("--- Starting Fast Lookup Phase (Serial) ---")
    all_rows = []
    input_df['Filing Date'] = pd.to_datetime(input_df['Filing Date'])
    for index, row in input_df.iterrows():
        ticker, target_date = row['Ticker'], row['Filing Date']
        stock_df = data_warehouse.get(ticker)
        spx_df = data_warehouse.get("^spx")
        vixy_df = data_warehouse.get("vixy.us")
        if stock_df is None or spx_df is None or vixy_df is None:
            continue
        try:
            stock_features = stock_df.loc[:target_date].iloc[-1]
            spx_features = spx_df.loc[:target_date].iloc[-1]
            vixy_features = vixy_df.loc[:target_date].iloc[-1]
        except IndexError:
            continue
        result_dict = stock_features.to_dict()
        result_dict['Days_Since_IPO'] = (target_date - stock_df.index.min()).days
        result_dict.update(spx_features.add_prefix('Market_SPX_').to_dict())
        result_dict.update(vixy_features.add_prefix('Market_VIXY_').to_dict())
        result_dict['Ticker'] = ticker
        result_dict['Filing Date'] = target_date
        all_rows.append(result_dict)

    print("--- ✅ Fast Lookup Phase Complete ---\n")
    if not all_rows:
        return pd.DataFrame()
        
    final_df = pd.DataFrame(all_rows)
    cols = ['Ticker', 'Filing Date'] + [col for col in final_df.columns if col not in ['Ticker', 'Filing Date']]
    return final_df[cols]


if __name__ == '__main__':
    DB_PATH = "../../../data/stooq_database" 
    OUTPUT_CSV = "technical_features_summary_fast.csv"
    
    # --- CHANGE: The test block no longer needs N_JOBS ---
    data = {
        "Ticker": ["AAPL", "MSFT", "GOOG", "TSLA", "AAPL", "MSFT", "GOOG", "TSLA"],
        "Filing Date": ["2024-01-10", "2023-12-29", "2024-04-15", "2024-03-20", 
                        "2023-05-10", "2023-06-29", "2023-07-15", "2023-08-20"]
    }
    input_df = pd.DataFrame(data)
    print("Input Data:")
    print(input_df)
    
    # The function call is now much cleaner
    techn_ind_df = load_technical_indicators_df(input_df, DB_PATH)

    if not techn_ind_df.empty:
        print(f"\n--- Final Combined Summary ---")
        print(f"Generated DataFrame with {techn_ind_df.shape[0]} rows and {techn_ind_df.shape[1]} columns.")
        techn_ind_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Summary successfully saved to '{OUTPUT_CSV}'")
    else:
        print("\nCould not calculate technical features for any input.")
