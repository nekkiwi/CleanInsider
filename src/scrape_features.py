# file: /src/scrape_features_parallel.py (Corrected for Multiprocessing)

# --- All imports should be at the top level ---

import time
from pathlib import Path

import pandas as pd

from scrapers.feature_scraper.feature_scraper_util.general_utils import (
    add_date_features,
    create_composite_features,
    report_missing_data,
)
from scrapers.feature_scraper.load_macro_features import load_macro_feature_df

# Your own module imports
from scrapers.feature_scraper.load_sec_data import load_sec_features_df
from scrapers.feature_scraper.load_technical_indicators import (
    load_technical_indicators_df,
)
from scrapers.feature_scraper.scrape_openinsider import scrape_openinsider

# --- All class and function definitions should be at the top level ---


def run_feature_scraping_pipeline(num_weeks: int, config):
    """
    Main function to run the entire feature scraping and engineering pipeline.
    """
    start_time = time.time()

    # --- Step 1: Scrape OpenInsider data ---
    print("--- Step 1: Scraping base insider data from OpenInsider ---")
    base_df = scrape_openinsider(num_weeks=num_weeks)
    if base_df.empty:
        print("No base data scraped from OpenInsider. Halting pipeline.")
        return
    base_df["Filing Date"] = pd.to_datetime(base_df["Filing Date"])

    # --- Step 2: Generate SEC financial features ---
    print("\n--- Step 2: Generating SEC financial features (in parallel) ---")
    sec_df = load_sec_features_df(
        input_df=base_df.copy(),
        parquet_dir_str=config.EDGAR_DOWNLOAD_PATH,
        request_header=config.REQUESTS_HEADER,
        n_prev=2,
    )

    # --- Step 5: Merge (Moved up to filter early) ---
    print("\n--- Merging SEC feature set ---")
    if not sec_df.empty:
        sec_df["Filing Date"] = pd.to_datetime(sec_df["Filing Date"])
        # Perform the merge
        merged_df = pd.merge(base_df, sec_df, on=["Ticker", "Filing Date"], how="left")

        # --- NEW: Drop tickers that were not found in SEC data ---
        # We use 'CIK' as a reliable indicator that the SEC merge was successful.
        rows_before_drop = len(merged_df)
        merged_df.dropna(subset=["CIK"], inplace=True)
        rows_after_drop = len(merged_df)

        print(
            f"   -> Dropped {rows_before_drop - rows_after_drop} records that were not found in SEC data."
        )
        if merged_df.empty:
            print("No records remain after filtering for SEC data. Halting.")
            return
    else:
        print(
            "   -> No SEC data was generated. The resulting feature set will not contain financial statement features."
        )
        merged_df = base_df

    # --- Step 3 & 4 (Now use the filtered DataFrame) ---
    print("\n--- Step 3: Generating technical indicator features (in parallel) ---")
    technical_df = load_technical_indicators_df(
        input_df=merged_df.copy(), db_path_str=config.STOOQ_DATABASE_PATH
    )
    if not technical_df.empty:
        technical_df["Filing Date"] = pd.to_datetime(technical_df["Filing Date"])
        merged_df = pd.merge(
            merged_df, technical_df, on=["Ticker", "Filing Date"], how="left"
        )

    print("\n--- Step 4: Generating macroeconomic features ---")
    dates_list = merged_df["Filing Date"].unique().tolist()
    macro_df = load_macro_feature_df(
        dates_list=dates_list, stooq_db_dir=config.STOOQ_DATABASE_PATH
    )
    if not macro_df.empty:
        macro_df = macro_df.rename(columns={"Query_Date": "Filing Date"})
        macro_df["Filing Date"] = pd.to_datetime(macro_df["Filing Date"])
        merged_df = pd.merge(merged_df, macro_df, on="Filing Date", how="left")

    cols_to_keep = [
        col
        for col in merged_df.columns
        if isinstance(col, str) and not col.startswith("Unnamed")
    ]
    # Then, select only these valid columns
    final_df = merged_df[cols_to_keep]
    print("   ✅ All feature sets successfully merged.")

    # --- Steps 6, 7, 8 ... ---
    print("\n--- Step 6: Engineering final composite features ---")
    final_df = create_composite_features(final_df)
    final_df = add_date_features(final_df)
    print("   ✅ Final features created.")

    print("\n--- Step 7: Final Data Quality Report ---")
    output_directory = Path(config.FEATURES_OUTPUT_PATH)
    report_missing_data(final_df, output_dir=output_directory)

    print(
        f"\n--- Step 7.5: Saving MERGED (pre-pruning) dataset to {config.RAW_FEATURES_PATH} ---"
    )
    final_df.to_parquet(config.RAW_FEATURES_PATH, index=False)

    end_time = time.time()
    print(f"\n--- ✅ Pipeline Complete in {end_time - start_time:.2f} seconds ---")
