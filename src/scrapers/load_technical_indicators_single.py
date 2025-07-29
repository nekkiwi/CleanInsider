# file: calculate_technical_indicators_fast.py (Optimized for Batch Processing)
import warnings

# hide pandas‑ta’s “Setting an item of incompatible dtype…” FutureWarning
warnings.simplefilter("ignore", FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

# Monkey-patch numpy to support pandas_ta expectations
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# only filter that specific deprecation warning from pandas_ta
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    module="pandas_ta"
)

import pandas_ta as ta

# --- Data Loading (Local Only) - This helper remains the same ---
STOOQ_COLUMN_MAP = {
    '<DATE>': 'Date',
    '<OPEN>': 'Open',
    '<HIGH>': 'High',
    '<LOW>': 'Low',
    '<CLOSE>': 'Close',
    '<VOL>': 'Volume'
}

@lru_cache(maxsize=256) # Increased cache size for more tickers
def find_and_load_ohlcv_data(db_path_str: str, ticker: str) -> pd.DataFrame:
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

# --- Feature Calculation - This helper also remains largely the same ---
def calculate_indicators(stock_df: pd.DataFrame, is_market_instrument=False) -> pd.DataFrame:
    if stock_df.empty: return pd.DataFrame()
    
    # 52-Week High/Low Calculation
    rolling_high_52w = stock_df['Close'].rolling(window=252, min_periods=1).max()
    rolling_low_52w = stock_df['Close'].rolling(window=252, min_periods=1).min()
    stock_df['Dist_52w_High_Pct'] = (stock_df['Close'] - rolling_high_52w) / rolling_high_52w
    stock_df['Dist_52w_Low_Pct'] = (stock_df['Close'] - rolling_low_52w) / rolling_low_52w

    # Define strategies based on whether it's the main stock or a market index
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
    
    # ---- NEW: ensure all columns are float64 to avoid dtype conflicts ----
    for col in stock_df.columns:
        if stock_df[col].dtype == 'int64':
            stock_df[col] = stock_df[col].astype('float64')
            
    return stock_df

# --- NEW: Pre-computation function ---
def preload_and_calculate_all_data(tickers: list, db_path_str: str) -> dict:
    """
    Loads and pre-computes all indicators for all required tickers once.
    This is the core of the optimization.
    """
    print("--- Starting Pre-computation Phase ---")
    data_warehouse = {}
    for ticker in tickers:
        print(f"   ▶️  Pre-loading and processing {ticker}...")
        df = find_and_load_ohlcv_data(db_path_str, ticker)
        if df.empty:
            print(f"   ❌ Could not load data for {ticker}. It will be skipped.")
            continue
        
        # Use the appropriate strategy for market vs. primary tickers
        is_market = ticker.lower() in ["^spx", "vixy.us"]
        indicators_df = calculate_indicators(df, is_market_instrument=is_market)
        
        # Store the fully computed DataFrame in our in-memory warehouse
        data_warehouse[ticker] = indicators_df
    
    print("--- ✅ Pre-computation Phase Complete ---\n")
    return data_warehouse

# --- REFACTORED: Main orchestrator for batch processing ---
def load_technical_indicators_df(input_df: pd.DataFrame, db_path_str: str) -> pd.DataFrame:
    """
    Takes a DataFrame and returns a summary with all technical features,
    using a highly optimized pre-computation strategy.
    """
    # 1. Get a unique list of all tickers we need to process
    unique_tickers = input_df['Ticker'].unique().tolist()
    # Always include the market instruments
    market_tickers = ["^spx", "vixy.us"]
    all_needed_tickers = sorted(list(set(unique_tickers + market_tickers)))

    # 2. Run the pre-computation step
    data_warehouse = preload_and_calculate_all_data(all_needed_tickers, db_path_str)

    # 3. Process each row using fast, in-memory lookups
    print("--- Starting Fast Lookup Phase ---")
    all_rows = []
    
    for index, row in input_df.iterrows():
        ticker, filing_date_str = row['Ticker'], row['Filing Date']
        target_date = pd.to_datetime(filing_date_str)
        
        # Get pre-computed DataFrames from the warehouse
        stock_df = data_warehouse.get(ticker)
        spx_df = data_warehouse.get("^spx")
        vixy_df = data_warehouse.get("vixy.us")

        if stock_df is None or spx_df is None or vixy_df is None:
            print(f"   ⚠️ Skipping {ticker} on {filing_date_str} due to missing pre-computed data.")
            continue

        # Perform the lookup. This is much faster than recalculating.
        try:
            stock_features = stock_df.loc[:target_date].iloc[-1]
            spx_features = spx_df.loc[:target_date].iloc[-1]
            vixy_features = vixy_df.loc[:target_date].iloc[-1]
        except IndexError:
            # This happens if there's no data on or before the target date
            print(f"   ⚠️ Skipping {ticker} on {filing_date_str} (no data available for date).")
            continue
            
        # Assemble the final dictionary for the row
        result_dict = stock_features.to_dict()
        result_dict['Days_Since_IPO'] = (target_date - stock_df.index.min()).days
        
        # Add market features with prefixes
        result_dict.update(spx_features.add_prefix('Market_SPX_').to_dict())
        result_dict.update(vixy_features.add_prefix('Market_VIXY_').to_dict())
        
        result_dict['Ticker'] = ticker
        result_dict['Filing Date'] = filing_date_str
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

    # For a large run, you would load your 100k entries here
    # For testing, we'll use a small sample with duplicate tickers
    data = {
        "Ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "Filing Date": ["2024-01-10", "2023-12-29", "2024-04-15", "2024-03-20"]
    }
    input_df = pd.DataFrame(data)
    print("Input Data:")
    print(input_df)
    
    techn_ind_df = load_technical_indicators_df(input_df, DB_PATH)

    if not techn_ind_df.empty:
        print(f"--- Final Combined Summary ---")
        print(f"Generated DataFrame with {techn_ind_df.shape[0]} rows and {techn_ind_df.shape[1]} columns.")
        techn_ind_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 Summary successfully saved to '{OUTPUT_CSV}'")
    else:
        print("\nCould not calculate technical features for any input.")
