# file: src/scrapers/target_scraper/calculate_master_targets.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yfinance as yf
import warnings
from joblib import Parallel, delayed

from src.scrapers.data_loader import load_ohlcv_with_fallback
from .generate_targets import calculate_realized_alpha_series


SPX_TICKER_LOCAL = '^spx'
SPX_TICKER_YFINANCE = '^GSPC'


def get_and_update_spx_data(config, min_required_date: pd.Timestamp):
    # This function remains correct and is unchanged.
    print("\n--- Pre-loading and Verifying SPX Market Data ---")
    local_spx_path = Path(config.STOOQ_DATABASE_PATH) / f"{SPX_TICKER_LOCAL}.parquet"
    try:
        spx_local = pd.read_parquet(local_spx_path)
        if spx_local.index.tz is not None:
             spx_local.index = spx_local.index.tz_localize(None)
    except FileNotFoundError:
        spx_local = pd.DataFrame()
    today = pd.Timestamp.now().normalize()
    download_needed = False
    download_start_date = min_required_date
    if spx_local.empty:
        print("-> Local SPX cache not found. A full download is required.")
        download_needed = True
        download_start_date = min_required_date
    else:
        is_history_sufficient = spx_local.index.min() <= min_required_date
        is_data_fresh = spx_local.index.max() >= today - pd.Timedelta(days=3)
        if is_history_sufficient and is_data_fresh:
            print(f"-> Local SPX data is complete and up-to-date (Covers {spx_local.index.min().date()} to {spx_local.index.max().date()}).")
            return spx_local
        elif not is_history_sufficient:
            print(f"-> Local SPX history is incomplete. Cache starts at {spx_local.index.min().date()}, but data is needed from {min_required_date.date()}.")
            download_needed = True
            download_start_date = min_required_date
            spx_local = pd.DataFrame() 
        elif not is_data_fresh:
            print(f"-> Local SPX data is stale. Latest point is {spx_local.index.max().date()}.")
            download_needed = True
            download_start_date = spx_local.index.max() + pd.Timedelta(days=1)
    if download_needed:
        print(f"-> Attempting to fetch/update SPX data from yfinance starting from {download_start_date.date()}...")
        try:
            spx_yf = yf.download(SPX_TICKER_YFINANCE, start=download_start_date, progress=False, auto_adjust=False)
            if not spx_yf.empty:
                print(f"-> Successfully downloaded {len(spx_yf)} new rows of SPX data.")
                spx_yf.index = spx_yf.index.tz_localize(None)
                updated_spx_data = pd.concat([spx_local, spx_yf])
                updated_spx_data = updated_spx_data[~updated_spx_data.index.duplicated(keep='last')]
                updated_spx_data.sort_index(inplace=True)
                local_spx_path.parent.mkdir(parents=True, exist_ok=True)
                updated_spx_data.to_parquet(local_spx_path)
                return updated_spx_data
            else:
                if spx_local.empty: return pd.DataFrame()
                else:
                    warnings.warn("Could not fetch fresh SPX data; proceeding with existing local data.")
                    return spx_local
        except Exception as e:
            print(f"-> Yfinance download for SPX failed: {e}")
            if spx_local.empty: return pd.DataFrame()
            else:
                warnings.warn("Yfinance download failed; proceeding with existing local data.")
                return spx_local
    return spx_local


# --- THIS IS THE NEW WORKER FUNCTION FOR PARALLEL DATA LOADING ---
def _load_ohlcv_for_ticker(ticker, required_start_date, db_path_str):
    """Worker function to load OHLCV data for a single ticker."""
    df = load_ohlcv_with_fallback(
        ticker, 
        db_path_str, 
        required_start_date=required_start_date
    )
    # Return the ticker along with the data for easy reconstruction
    return ticker, df


def calculate_master_targets(config, target_combinations: list, batch_size: int = 100, debug: bool = False):
    print("\n--- STEP 2: Calculating Master Targets in Batches ---")
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)
    event_list_path = targets_dir / "master_event_list.parquet"
    output_path = targets_dir / "master_targets.parquet"
    if not event_list_path.exists():
        raise FileNotFoundError(f"Master event list not found at {event_list_path}.")
    
    base_df = pd.read_parquet(event_list_path)
    base_df['Filing Date'] = pd.to_datetime(base_df['Filing Date']).dt.tz_localize(None)
    all_tickers = base_df['Ticker'].unique()
    
    if base_df.empty:
        raise ValueError("Master event list is empty.")
    
    min_date_needed = base_df['Filing Date'].min() - pd.Timedelta(days=90)
    spx_data = get_and_update_spx_data(config, min_date_needed)
    
    if spx_data.empty:
        raise RuntimeError("FATAL: Could not load SPX data from any source.")
    if spx_data.index.min() > min_date_needed:
        raise RuntimeError(f"FATAL: Final SPX data does not cover the required historical range.")
        
    print(f"--- ✅ SPX data ready (Range: {spx_data.index.min().date()} to {spx_data.index.max().date()}) ---\n")

    all_results = []
    for i in tqdm(range(0, len(all_tickers), batch_size), desc="Processing Ticker Batches"):
        ticker_batch = all_tickers[i:i + batch_size]
        batch_df = base_df[base_df['Ticker'].isin(ticker_batch)].copy()
        
        # --- PARALLEL OHLCV DATA LOADING LOGIC ---
        # 1. Prepare the tasks for all tickers in the current batch
        tasks = []
        for ticker, events_for_ticker in batch_df.groupby('Ticker'):
            min_filing_date_for_ticker = events_for_ticker['Filing Date'].min()
            required_start_date = min_filing_date_for_ticker - pd.Timedelta(days=90)
            tasks.append(
                delayed(_load_ohlcv_for_ticker)(ticker, required_start_date, config.STOOQ_DATABASE_PATH)
            )

        # 2. Execute the data loading in parallel for the batch
        parallel_ohlcv_results = Parallel(n_jobs=-2)(
            tqdm(tasks, desc="Loading price data for batch", leave=False, total=len(tasks))
        )
        
        # 3. Reconstruct the ohlcv_data dictionary from the parallel results
        ohlcv_data = {ticker: df for ticker, df in parallel_ohlcv_results if df is not None and not df.empty}
        # --- END OF PARALLEL LOGIC ---

        batch_results_df = batch_df[['Ticker', 'Filing Date']].copy()
        for params in target_combinations:
            timepoint, tp, sl = params['time'], params['tp'], params['sl']
            col_name = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"
            
            # This function is already parallel internally, so we call it as before
            alpha_series, debug_log = calculate_realized_alpha_series(
                base_df=batch_df, ohlcv_data=ohlcv_data, spx_data=spx_data,
                timepoint_str=timepoint, take_profit=tp, stop_loss=sl,
                debug=debug
            )
            batch_results_df[col_name] = alpha_series
            
            if debug and debug_log:
                print(f"\n--- Debug Log for {col_name} (Batch {i//batch_size + 1}) ---")
                for msg in debug_log: print(msg)
                print("--------------------------------------------------\n")
        
        all_results.append(batch_results_df)

    if all_results:
        print("\nCombining results from all batches...")
        final_df = pd.concat(all_results, ignore_index=True)
        print(f"Saving final master targets file to {output_path}")
        final_df.to_parquet(output_path, engine='pyarrow', index=False)
        print("✅ Master target calculations complete.")
    else:
        print("No results were generated to save.")

