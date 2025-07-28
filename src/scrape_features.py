import pandas as pd
from .scrapers.scrape_openinsider import scrape_openinsider
from .scrapers.scrape_technical_indicators import scrape_all_technical_indicators
from .scrapers.scrape_financial_ratios import scrape_all_financial_ratios
from .scrapers.scrape_feature_utils.general_utils import merge_and_save_features, report_missing_data, create_composite_features

def run_feature_scraping_pipeline(num_weeks: int, config):
    """Orchestrates the entire feature scraping and engineering pipeline."""
    # Step 1: Get base data from OpenInsider (already feature-engineered)
    base_df = scrape_openinsider(num_weeks=num_weeks, config=config)
    if base_df.empty:
        print("Scraping OpenInsider failed or returned no data. Aborting pipeline.")
        return
    
    # Step 2: Scrape technical, market, and date features
    scrape_all_technical_indicators(base_df=base_df, config=config)
    
    # Step 3: Scrape financial ratio and sector features
    scrape_all_financial_ratios(base_df=base_df, config=config)
    
    # Step 4: Merge all feature sets
    merged_df = merge_and_save_features(config=config)
    if merged_df.empty:
        print("Merging failed or resulted in empty data. Aborting.")
        return
        
    # Step 5: Create composite features from the merged data
    final_df = create_composite_features(merged_df)

    # Step 6: Save the final, complete dataset and report quality
    final_df.to_excel(config.FINAL_OUTPUT_PATH, index=False)
    print(f"Final merged and enriched data saved to: {config.FINAL_OUTPUT_PATH}\n")
    report_missing_data(final_df)
