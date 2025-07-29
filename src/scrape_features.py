import pandas as pd
from .scrapers.scrape_openinsider import scrape_openinsider
from .scrapers.scrape_technical_indicators import scrape_all_technical_indicators
from .scrapers.scrape_financial_ratios import scrape_all_financial_ratios
from .scrapers.scrape_feature_utils.general_utils import create_composite_features

def run_feature_scraping_pipeline(num_weeks: int, config):
    """
    Main function to run the entire feature scraping and engineering pipeline.
    This version passes explicit path arguments to the scraper functions.
    """
    # --- Step 1: Scrape base insider data ---
    stage_1_path = config.STAGE_1_PATH
    
    # **FIX: Unpack the tuple returned by scrape_openinsider.**
    # This assigns the DataFrame to `base_df` and discards the second element (the path).
    base_df, _ = scrape_openinsider(num_weeks=num_weeks, output_path=stage_1_path)
    
    # Now, this check will work correctly on the DataFrame.
    if base_df.empty:
        print("No base data scraped from OpenInsider. Halting pipeline.")
        return

    # --- Step 2: Scrape technical indicators ---
    stage_2_path = config.STAGE_2_PATH
    stooq_db_path = config.STOOQ_DATABASE_PATH
    scrape_all_technical_indicators(
        base_df=base_df, 
        output_path=stage_2_path,
        stooq_path=stooq_db_path
    )

    # --- Step 3: Scrape financial ratios ---
    stage_3_path = config.STAGE_3_PATH
    edgar_db_path = config.EDGAR_DOWNLOAD_PATH
    scrape_all_financial_ratios(
        base_df=base_df,
        output_path=stage_3_path,
        edgar_data_path=edgar_db_path
    )

    # --- Step 4: Merge all data sources ---
    print("\nMerging all data sources...")
    try:
        df1 = pd.read_excel(stage_1_path, parse_dates=['Filing Date'])
        df2 = pd.read_excel(stage_2_path, parse_dates=['Filing Date'])
        df3 = pd.read_excel(stage_3_path, parse_dates=['Filing Date'])
        
        final_df = pd.merge(df1, df2, on=['Ticker', 'Filing Date'], how='left')
        final_df = pd.merge(final_df, df3, on=['Ticker', 'Filing Date'], how='left')
        final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]

    except FileNotFoundError as e:
        print(f"Error: A required data file is missing. {e}")
        return
    except Exception as e:
        print(f"An error occurred during the merge process: {e}")
        return

    # --- Step 5: Create post-merge composite features ---
    print("Creating composite features...")
    final_df = create_composite_features(final_df)

    # --- Step 6: Save the final, complete dataset ---
    final_output_path = config.FINAL_OUTPUT_PATH
    final_df.to_excel(final_output_path, index=False)
    print(f"\n--- Pipeline Complete ---")
    print(f"Final feature dataset saved to: {final_output_path}")
    print(f"Total features created: {len(final_df.columns)}")
    print(f"Final dataset shape: {final_df.shape}")