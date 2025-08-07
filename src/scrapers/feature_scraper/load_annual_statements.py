# file: src/scrapers/feature_scraper/load_annual_statements.py

import pandas as pd
import yfinance as yf
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import time
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from threading import Lock

# Import SEC data loading functions
from .load_sec_data import load_sec_features_df

# Ignore common warnings from yfinance and pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_annual_data_for_ticker(ticker_symbol: str, sec_parquet_dir: str = None, request_header: str = None) -> pd.Series | None:
    """Fetches and processes annual data for a single ticker with SEC fallback."""
    try:
        # A small sleep is respectful to the API when running many requests
        time.sleep(0.01)
        ticker = yf.Ticker(ticker_symbol)
        
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow

        # More flexible check - we need at least one statement with at least 2 years of data
        available_statements = []
        if not financials.empty and financials.shape[1] >= 2:
            available_statements.append(financials)
        if not balance_sheet.empty and balance_sheet.shape[1] >= 2:
            available_statements.append(balance_sheet)
        if not cash_flow.empty and cash_flow.shape[1] >= 2:
            available_statements.append(cash_flow)

        if not available_statements:
            # Try SEC EDGAR fallback if available
            if sec_parquet_dir and request_header:
                try:
                    # Create a dummy DataFrame with the ticker to use SEC lookup
                    dummy_df = pd.DataFrame({
                        'Ticker': [ticker_symbol],
                        'Filing Date': [pd.Timestamp.now().strftime('%Y-%m-%d')]
                    })
                    sec_data = load_sec_features_df(dummy_df, sec_parquet_dir, request_header)
                    if not sec_data.empty:
                        # Convert SEC data to the same format as yfinance data
                        sec_series = sec_data.iloc[0].drop(['Ticker', 'Filing Date', 'CIK'])
                        sec_series['Ticker'] = ticker_symbol
                        return sec_series
                except Exception as e:
                    # Silently fail SEC fallback to reduce clutter
                    pass
            
            return None

        # Use available statements
        all_statements = pd.concat(available_statements)
        all_statements = all_statements.loc[~all_statements.index.duplicated(keep='first')]

        recent_data = all_statements.iloc[:, :2]
        recent_data.columns = ['Y1', 'Y2']
        
        features_y1 = recent_data['Y1'].rename(lambda x: f"FIN_{x}_Y1")
        features_y2 = recent_data['Y2'].rename(lambda x: f"FIN_{x}_Y2")
        features_diff = (recent_data['Y1'] - recent_data['Y2']).rename(lambda x: f"FIN_{x}_diff_Y1_Y2")
        
        final_features = pd.concat([features_y1, features_y2, features_diff])
        final_features['Ticker'] = ticker_symbol
        
        return final_features

    except Exception as e:
        # Only show rate limit errors, suppress others to reduce clutter
        if "YFRateLimitError" in str(e) or "Too Many Requests" in str(e):
            print(f"  [RATE_LIMIT] {ticker_symbol}: Rate limited by Yahoo Finance API")
        return None

def _process_ticker_batch_in_parallel(ticker_batch: list[str], sec_parquet_dir: str = None, request_header: str = None) -> list[pd.Series]:
    """
    Processes a batch of tickers in parallel with a thread-safe progress bar.
    Uses threading backend to avoid serialization issues with tqdm.
    """
    n_jobs = -2
    lock = Lock()
    results = []

    with tqdm(total=len(ticker_batch), desc="Processing Tickers", leave=False) as pbar:
        def wrapped(ticker):
            result = get_annual_data_for_ticker(ticker, sec_parquet_dir, request_header)
            with lock:
                pbar.update(1)
            return result

        results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(wrapped)(ticker) for ticker in ticker_batch)

    return [r for r in results if r is not None]

def generate_annual_statements(base_df: pd.DataFrame, output_path: str, batch_size: int = 100, missing_thresh: float = 0.8, sec_parquet_dir: str = None, request_header: str = None):
    """
    Orchestrates the download of annual financial data by processing
    batches sequentially, but processing tickers within each batch in parallel.
    """
    print("\n--- Generating Annual Statement Features (Sequential Batches, Parallel Tickers) ---")
    
    # Get unique Ticker-Filing Date combinations instead of just unique tickers
    unique_combinations = base_df[['Ticker', 'Filing Date']].drop_duplicates()
    unique_tickers = unique_combinations['Ticker'].unique().tolist()
    
    ticker_batches = [unique_tickers[i:i + batch_size] for i in range(0, len(unique_tickers), batch_size)]
    print(f"  > Divided {len(unique_tickers)} unique tickers into {len(ticker_batches)} batches of size {batch_size}.")
    print(f"  > Processing {len(unique_combinations)} unique Ticker-Filing Date combinations.")

    all_rows = []
    
    # --- THIS IS THE FIX ---
    # Use a standard for loop to iterate through the batches sequentially.
    # The outer tqdm tracks the progress of the BATCHES.
    for i, batch in enumerate(tqdm(ticker_batches, desc="Processing Annual Statements", unit="batch")):
        # Call the parallel processing function for the current batch
        batch_results = _process_ticker_batch_in_parallel(batch, sec_parquet_dir, request_header)
        all_rows.extend(batch_results)
        
        # Show progress every few batches
        if (i + 1) % 10 == 0:
            print(f"  > Processed {len(all_rows)} tickers so far...")
    # --- END OF FIX ---
    
    if not all_rows:
        print("❌ No valid annual statement data could be fetched. An empty file will be created.")
        pd.DataFrame().to_parquet(output_path, index=False)
        return

    annual_statements_df = pd.DataFrame(all_rows)

    # Now we need to expand this to create a row for each unique Ticker-Filing Date combination
    # First, merge with the unique combinations to get all the filing dates
    expanded_df = pd.merge(unique_combinations, annual_statements_df, on='Ticker', how='inner')
    
    print(f"  > Original component shape: {annual_statements_df.shape}")
    print(f"  > Expanded to {len(expanded_df)} rows for all Ticker-Filing Date combinations")
    
    # --- NEW: Pre-filter sparse columns ---
    missing_proportions = expanded_df.isnull().sum() / len(expanded_df)
    cols_to_drop = missing_proportions[missing_proportions >= missing_thresh].index
    expanded_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    print(f"  > Dropped {len(cols_to_drop)} columns with >= {missing_thresh:.0%} missing values.")
    print(f"  > Final component shape: {expanded_df.shape}")
    # --- END NEW ---

    expanded_df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(expanded_df)} records with financial data to {output_path}")
