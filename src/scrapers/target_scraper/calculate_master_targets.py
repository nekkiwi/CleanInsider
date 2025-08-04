# file: src/scrapers/target_scraper/calculate_master_targets.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq

from src.scrapers.data_loader import load_ohlcv_with_fallback
from .generate_targets import calculate_realized_alpha_series

SPX_TICKER_LOCAL = '^spx'
SPX_TICKER_YFINANCE = '^GSPC'

def calculate_master_targets(config, target_combinations: list, batch_size: int = 100, debug: bool = False):
    """
    Calculates all target combinations in batches, with a definitive fix for
    SPX data handling in parallel processing.
    """
    print("\n--- STEP 2: Calculating Master Targets in Batches ---")
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)
    event_list_path = targets_dir / "master_event_list.parquet"
    output_path = targets_dir / "master_targets.parquet"

    if not event_list_path.exists():
        raise FileNotFoundError(f"Master event list not found at {event_list_path}. Please run step 1 first.")

    base_df = pd.read_parquet(event_list_path)
    base_df['Filing Date'] = pd.to_datetime(base_df['Filing Date']).dt.tz_localize(None)
    all_tickers = base_df['Ticker'].unique()
    
    # --- THE FIX: Load and prepare SPX data ONCE before the loop ---
    print("\n--- Pre-loading and verifying SPX market data ---")
    spx_data = load_ohlcv_with_fallback(SPX_TICKER_LOCAL, config.STOOQ_DATABASE_PATH)
    
    use_fresh_spx = False
    if spx_data.empty:
        print("   -> Local SPX data not found.")
        use_fresh_spx = True
    elif spx_data.index.max() < pd.Timestamp.now().normalize() - pd.Timedelta(days=1):
        print(f"   -> Warning: Local SPX data is outdated (latest: {spx_data.index.max().date()}).")
        use_fresh_spx = True

    if use_fresh_spx:
        print("      -> Attempting to fetch fresh SPX data from yfinance...")
        try:
            start_date_needed = base_df['Filing Date'].min() - pd.Timedelta(days=1)
            print(f"         Requesting SPX data starting from: {start_date_needed.date()}")

            # Pass the 'start' parameter to the download call.
            spx_yf = yf.download(
                SPX_TICKER_YFINANCE, 
                start=start_date_needed, 
                progress=False, 
                auto_adjust=False
            )
            
            if not spx_yf.empty:
                print("      -> Successfully downloaded fresh SPX data.")
                spx_yf.index = spx_yf.index.tz_localize(None)
                spx_data = spx_yf
            else:
                print("      -> Yfinance download for SPX failed (returned empty).")
        except Exception as e:
            print(f"      -> Yfinance download for SPX failed with error: {e}")
    
    if spx_data.empty:
        raise RuntimeError("Could not load SPX data from any source. Halting pipeline.")
    
    print(f"--- ✅ SPX data ready (Range: {spx_data.index.min().date()} to {spx_data.index.max().date()}) ---\n")

    for i in tqdm(range(0, len(all_tickers), batch_size), desc="Processing Ticker Batches"):
        ticker_batch = all_tickers[i:i + batch_size]
        batch_df = base_df[base_df['Ticker'].isin(ticker_batch)]
        
        print(f"\nProcessing batch {i//batch_size + 1}: Loading data for {len(ticker_batch)} tickers...")
        ohlcv_data = {}
        # The batch loader no longer needs to worry about SPX
        for ticker in tqdm(ticker_batch, desc="Loading price data for batch"):
            df = load_ohlcv_with_fallback(ticker, config.STOOQ_DATABASE_PATH)
            if not df.empty:
                ohlcv_data[ticker] = df

        # The calculation now uses the definitive, pre-loaded spx_data
        batch_results_df = batch_df[['Ticker', 'Filing Date']].copy()
        for params in target_combinations:
            timepoint, tp, sl = params['time'], params['tp'], params['sl']
            col_name = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"
            
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
        
        # Parquet append logic remains the same
        if i == 0:
            batch_results_df.to_parquet(output_path, engine='pyarrow', index=False)
        else:
            # Append to existing parquet file
            existing_table = pq.read_table(output_path)
            new_table = pa.Table.from_pandas(batch_results_df, schema=existing_table.schema)
            with pq.ParquetWriter(output_path, existing_table.schema) as writer:
                writer.write_table(existing_table)
                writer.write_table(new_table)

    print(f"✅ Master target calculations complete. Final results saved to {output_path}")

