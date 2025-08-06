import pandas as pd
import numpy as np
import ta
from tqdm import tqdm
from joblib import Parallel, delayed
from ..data_loader import load_ohlcv_with_fallback
import warnings

def calculate_indicators(stock_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Calculates a curated set of technical indicators with extensive debugging.
    """
    # print(f"\n[CALC-INFO {ticker}] Entering calculate_indicators function.")
    warnings.filterwarnings("ignore", category=FutureWarning, module="ta.*")
    
    if stock_df.empty or len(stock_df) < 50:
        # print(f"[CALC-WARN {ticker}] DataFrame is empty or too short (len: {len(stock_df)}). Aborting calculation.")
        return pd.DataFrame() 

    # print(f"[CALC-DEBUG {ticker}] Initial raw data received. Shape: {stock_df.shape}. Columns: {stock_df.columns.to_list()}")

    df = stock_df.copy()
    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    # --- Data Cleaning and Validation ---
    # print(f"[CALC-DEBUG {ticker}] Starting data cleaning and validation.")
    try:
        # Check which required columns we actually have
        available_cols = [col for col in required_cols if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # print(f"[CALC-WARN {ticker}] Missing columns: {missing_cols}")
            # If Volume is missing, create it with zeros
            if 'Volume' in missing_cols and len(missing_cols) == 1:
                df['Volume'] = 0
                available_cols.append('Volume')
                # print(f"[CALC-INFO {ticker}] Created Volume column with zeros")
            else:
                print(f"[CALC-ERROR {ticker}] Too many missing required columns: {missing_cols}")
                return pd.DataFrame()
        
        # Convert available columns to numeric
        for col in available_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # print(f"[CALC-DEBUG {ticker}] Successfully converted OHLCV columns to numeric.")
    except Exception as e:
        print(f"\n\n--- [CRITICAL ERROR in calculate_indicators for {ticker}] ---")
        print(f"Error during numeric conversion: {e}")
        print("DataFrame Info right before crash:")
        print(f"Shape: {df.shape}, Columns: {df.columns.tolist()}")
        if not df.empty:
            print(f"DataFrame Head:\n{df.head().to_string()}")
        print("------------------------------------------------------\n\n")
        return pd.DataFrame()
    
    # Clean infinite values and handle NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Use pandas fillna method instead of deprecated method parameter
    df = df.ffill().bfill()
    
    # Only drop rows where Close is NaN (most critical column)
    df = df.dropna(subset=['Close'])
    
    # print(f"[CALC-DEBUG {ticker}] Data after cleaning. Shape: {df.shape}")

    if df.empty or len(df) < 20:
        print(f"[CALC-WARN {ticker}] DataFrame is empty or too short after cleaning (len: {len(df)}). Aborting calculation.")
        return pd.DataFrame()

    # --- TA Library Calculations ---
    try:
        # print(f"[CALC-DEBUG {ticker}] Applying 'ta' library functions...")
        
        # Add each category separately with better error handling
        try:
            df = ta.add_volume_ta(df, high="High", low="Low", close="Close", volume="Volume", fillna=True)
        except Exception as e:
            print(f"[CALC-WARN {ticker}] Volume TA calculation failed: {e}")
        
        try:
            df = ta.add_volatility_ta(df, high="High", low="Low", close="Close", fillna=True)
        except Exception as e:
            print(f"[CALC-WARN {ticker}] Volatility TA calculation failed: {e}")
        
        try:
            df = ta.add_trend_ta(df, high="High", low="Low", close="Close", fillna=True)
        except Exception as e:
            print(f"[CALC-WARN {ticker}] Trend TA calculation failed: {e}")
        
        try:
            df = ta.add_momentum_ta(df, high="High", low="Low", close="Close", volume="Volume", fillna=True)
        except Exception as e:
            print(f"[CALC-WARN {ticker}] Momentum TA calculation failed: {e}")
        
        # print(f"[CALC-DEBUG {ticker}] 'ta' functions applied. Shape is now: {df.shape}")
        
    except Exception as e:
        print(f"  [CALC-ERROR {ticker}] TA calculation failed: {e}")
        # Don't return empty, use original data if TA fails
        # print(f"  [CALC-WARN {ticker}] Continuing with original OHLCV data only")

    # --- Manual Indicator Calculations ---
    try:
        # print(f"[CALC-DEBUG {ticker}] Calculating 52-week high/low distance...")
        
        # Ensure we have enough data for 52-week calculations
        min_periods = min(50, len(df) // 2)
        window_size = min(252, len(df))
        
        rolling_high_52w = df["Close"].rolling(window=window_size, min_periods=min_periods).max()
        rolling_low_52w = df["Close"].rolling(window=window_size, min_periods=min_periods).min()
        
        # Avoid division by zero
        df["Dist_52w_High_Pct"] = np.where(rolling_high_52w > 0, 
                                          (df["Close"] - rolling_high_52w) / rolling_high_52w, 
                                          0)
        df["Dist_52w_Low_Pct"] = np.where(rolling_low_52w > 0, 
                                         (df["Close"] - rolling_low_52w) / rolling_low_52w, 
                                         0)
        
        # print(f"[CALC-DEBUG {ticker}] 52-week calculations completed")
        
    except Exception as e:
        print(f"  [CALC-WARN {ticker}] Manual 52-week calculation failed: {e}")

    # print(f"[CALC-SUCCESS {ticker}] Finished indicator calculation. Final shape: {df.shape}")
    return df

def _process_ticker_for_technicals(work_item: tuple, db_path_str: str) -> list[dict] | None:
    """Worker function to process all events for a single ticker with verbose logging."""
    ticker, filing_dates = work_item
    
    min_date = pd.to_datetime(min(filing_dates)) - pd.Timedelta(days=365*2)
    
    # print(f"\n[WORKER-INFO {ticker}] Starting processing for {len(filing_dates)} filing dates. Need history from {min_date.date()}.")
    stock_df_raw = load_ohlcv_with_fallback(ticker, db_path_str, required_start_date=min_date)
    
    if stock_df_raw.empty: 
        print(f"[WORKER-FAIL {ticker}] No OHLCV data found from any source. Skipping ticker.")
        return None
    
    stock_df_indicators = calculate_indicators(stock_df_raw, ticker)
    if stock_df_indicators.empty: 
        print(f"[WORKER-FAIL {ticker}] Indicator calculation resulted in an empty DataFrame. Skipping ticker.")
        return None

    ticker_rows = []
    # print(f"[WORKER-INFO {ticker}] Looking up technical features for each filing date...")
    
    for date_str in filing_dates:
        target_date = pd.to_datetime(date_str)
        try:
            # Use asof to get the most recent data before or on the target date
            if target_date in stock_df_indicators.index:
                features = stock_df_indicators.loc[target_date]
            else:
                # Find the closest date before target_date
                available_dates = stock_df_indicators.index[stock_df_indicators.index <= target_date]
                if len(available_dates) > 0:
                    closest_date = available_dates.max()
                    features = stock_df_indicators.loc[closest_date]
                else:
                    print(f"  [WORKER-WARN {ticker}] No data available for or before {target_date.date()}")
                    continue
            
            # Convert to dictionary and add metadata
            if isinstance(features, pd.Series):
                result_dict = features.to_dict()
                result_dict["Ticker"] = ticker
                result_dict["Filing Date"] = date_str  # Standardize to Filing_Date in output
                ticker_rows.append(result_dict)
                # print(f"  [WORKER-DEBUG {ticker}] Found features for {target_date.date()}")
            else:
                print(f"  [WORKER-WARN {ticker}] Unexpected data type for features: {type(features)}")
                
        except Exception as e:
            print(f"  [WORKER-ERROR {ticker}] Error processing date {target_date.date()}: {e}")
            continue
            
    # print(f"[WORKER-SUCCESS {ticker}] Successfully processed {len(ticker_rows)} of {len(filing_dates)} events.")
    return ticker_rows if ticker_rows else None


def generate_technical_indicators(base_df: pd.DataFrame, db_path_str: str, output_path: str, missing_thresh: float = 0.8):
    """Orchestrates technical indicator calculation with comprehensive logging."""
    print("\n--- Generating Technical Indicator Features ---")
    print(f"[TECH-INFO] Input base_df shape: {base_df.shape}")
    print(f"[TECH-INFO] Available columns: {base_df.columns.tolist()}")
    
    if base_df.empty:
        print("❌ Base DataFrame is empty. Cannot generate technical indicators.")
        pd.DataFrame().to_parquet(output_path, index=False)
        return
    
    # Find correct column names
    ticker_col = next((col for col in base_df.columns if col.lower() in ['ticker', 'symbol']), None)
    date_col = next((col for col in base_df.columns if 'filing' in col.lower() or 'date' in col.lower()), None)
    
    if not ticker_col or not date_col:
        print(f"❌ Could not find required columns. Ticker: {ticker_col}, Date: {date_col}")
        pd.DataFrame().to_parquet(output_path, index=False)
        return
    
    print(f"[TECH-INFO] Using ticker column: '{ticker_col}' and date column: '{date_col}'")
    
    # Create work items
    work_items = list(base_df.groupby(ticker_col)[date_col].apply(list).items())
    print(f"[TECH-INFO] Created {len(work_items)} work items for {base_df[ticker_col].nunique()} unique tickers")
    
    # Process in parallel
    n_jobs = max(1, min(4, len(work_items)))
    tasks = [delayed(_process_ticker_for_technicals)(item, db_path_str) for item in work_items]
    results = Parallel(n_jobs=n_jobs)(tqdm(tasks, desc="Calculating Technicals"))
    
    # Combine results
    all_rows = []
    successful_tickers = 0
    failed_tickers = 0
    
    for i, ticker_rows in enumerate(results):
        if ticker_rows:
            all_rows.extend(ticker_rows)
            successful_tickers += 1
        else:
            failed_tickers += 1
    
    print(f"\n[TECH-INFO] Processing complete:")
    print(f"  - Successful tickers: {successful_tickers}")
    print(f"  - Failed tickers: {failed_tickers}")
    print(f"  - Total feature rows generated: {len(all_rows)}")
    
    if not all_rows:
        print("\n❌ No technical indicators could be generated.")
        pd.DataFrame().to_parquet(output_path, index=False)
        return
    
    final_df = pd.DataFrame(all_rows)
    print(f"[TECH-INFO] Combined dataframe shape: {final_df.shape}")
    print(f"[TECH-DEBUG] Sample columns: {final_df.columns[:10].tolist()}...")
    
    # Remove OHLCV columns
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    before_removal = final_df.shape[1]
    final_df.drop(columns=ohlcv_cols, inplace=True, errors="ignore")
    removed_cols = before_removal - final_df.shape[1]
    print(f"[TECH-DEBUG] Removed {removed_cols} OHLCV columns")
    
    # Apply missing value filter
    print(f"\n[TECH-INFO] Applying missing value filter (threshold: {missing_thresh})")
    if len(final_df) > 0:
        missing_proportions = final_df.isnull().sum() / len(final_df)
        cols_to_drop = missing_proportions[missing_proportions >= missing_thresh].index.tolist()
        
        if cols_to_drop:
            print(f"[TECH-DEBUG] Dropping {len(cols_to_drop)} columns with >= {missing_thresh*100:.0f}% missing:")
            print(f"[TECH-DEBUG] Dropped columns: {cols_to_drop[:10]}{'...' if len(cols_to_drop) > 10 else ''}")
            final_df.drop(columns=cols_to_drop, inplace=True)
        
        print(f"[TECH-INFO] Final shape after filtering: {final_df.shape}")
    
    final_df.to_parquet(output_path, index=False)
    print(f"\n✅ Saved {len(final_df)} technical feature records to {output_path}")
