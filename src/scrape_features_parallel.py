# file: /src/scrape_features_parallel.py (Corrected for Multiprocessing)

# --- All imports should be at the top level ---
import pandas as pd
from pathlib import Path
import time
import traceback

# Your own module imports
from scrapers.load_sec_data import load_sec_features_df
from scrapers.load_technical_indicators import load_technical_indicators_df
from scrapers.load_macro_features import load_macro_feature_df
from scrapers.scrape_openinsider import scrape_openinsider
from scrapers.scrape_feature_utils.general_utils import (
    create_composite_features,
    add_date_features,
    report_missing_data
)

# --- All class and function definitions should be at the top level ---

class Config:
    EDGAR_DOWNLOAD_PATH = "../data/sec_database/parquet"
    STOOQ_DATABASE_PATH = "../data/stooq_database"
    FINAL_OUTPUT_PATH = "../data/final_dataset.parquet"
    USER_AGENT = "your.name@yourdomain.com"

def run_feature_scraping_pipeline(num_weeks: int, config):
    """
    Main function to run the entire feature scraping and engineering pipeline.
    The sub-modules now manage their own parallelism.
    """
    start_time = time.time()
    
    # --- Step 1 is unchanged ---
    print("--- Step 1: Scraping base insider data from OpenInsider ---")
    base_df = scrape_openinsider(num_weeks=num_weeks)
    if base_df.empty:
        print("No base data scraped from OpenInsider. Halting pipeline.")
        return
    print(f"   → Scraped {len(base_df)} initial records.")
    base_df['Filing Date'] = pd.to_datetime(base_df['Filing Date'])
    
    # --- Step 2 & 3: Calls are now simpler ---
    print("\n--- Step 2: Generating SEC financial features (in parallel) ---")
    sec_df = load_sec_features_df(
        input_df=base_df.copy(),
        parquet_dir_str=config.EDGAR_DOWNLOAD_PATH,
        user_agent=config.USER_AGENT,
        n_prev=2
    )
    if not sec_df.empty:
        sec_df['Filing Date'] = pd.to_datetime(sec_df['Filing Date'])

    print("\n--- Step 3: Generating technical indicator features (in parallel) ---")
    technical_df = load_technical_indicators_df(
        input_df=base_df.copy(),
        db_path_str=config.STOOQ_DATABASE_PATH
    )
    if not technical_df.empty:
        technical_df['Filing Date'] = pd.to_datetime(technical_df['Filing Date'])

    # # --- Step 4: Macro features ---
    print("\n--- Step 4: Generating macroeconomic features ---")
    dates_list = base_df['Filing Date'].unique().tolist()
    macro_df = load_macro_feature_df(
        dates_list=dates_list,
        stooq_db_dir=config.STOOQ_DATABASE_PATH
    )
    macro_df = macro_df.rename(columns={'Query_Date': 'Filing Date'})
    macro_df['Filing Date'] = pd.to_datetime(macro_df['Filing Date'])

    # --- Step 5: Merge ---
    print("\n--- Step 5: Merging all feature sets ---")
    final_df = base_df
    if not sec_df.empty:
        final_df = pd.merge(final_df, sec_df, on=['Ticker', 'Filing Date'], how='left')
    if not technical_df.empty:
        final_df = pd.merge(final_df, technical_df, on=['Ticker', 'Filing Date'], how='left')
    if not macro_df.empty:
        final_df = pd.merge(final_df, macro_df, on='Filing Date', how='left')
    final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]
    print("   ✅ All feature sets successfully merged.")
    
    # --- Steps 6, 7, 8 ... ---
    print("\n--- Step 6: Engineering final composite features ---")
    final_df = create_composite_features(final_df)
    final_df = add_date_features(final_df)
    print("   ✅ Final features created.")
    
    print("\n--- Step 7: Final Data Quality Report ---")
    report_missing_data(final_df)

    final_output_path = Path(config.FINAL_OUTPUT_PATH)
    print(f"\n--- Step 8: Saving final dataset to {final_output_path} ---")
    final_df.to_parquet(final_output_path, index=False)
    
    end_time = time.time()
    print(f"\n--- ✅ Pipeline Complete in {end_time - start_time:.2f} seconds ---")
    print(f"Final dataset shape: {final_df.shape}")


def main():
    """
    Main entry point for the script.
    """
    try:
        NUM_WEEKS_TO_SCRAPE = 3
        config = Config()
        
        print(f"🚀 --- Starting Feature Scraping Pipeline (Parallelism managed by sub-modules) --- 🚀\n")

        run_feature_scraping_pipeline(
            num_weeks=NUM_WEEKS_TO_SCRAPE,
            config=config
        )
    except Exception as e:
        print(f"\n--- ❌ A critical error occurred in the pipeline: {e} ---")
        traceback.print_exc()
        
if __name__ == '__main__':
    main()