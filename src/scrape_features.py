# file: src/scrape_features.py

import pandas as pd
from pathlib import Path
import time
from src.scrapers.feature_scraper.scrape_openinsider import scrape_openinsider
from src.scrapers.feature_scraper.load_technical_indicators import generate_technical_indicators
from src.scrapers.feature_scraper.load_annual_statements import generate_annual_statements
from src.scrapers.feature_scraper.build_event_ohlcv import build_event_ohlcv_datasets

def run_feature_scraping_pipeline(num_weeks: int, config):
    """
    Orchestrates the new, component-based feature scraping pipeline.
    """
    start_time = time.time()
    
    # Define paths for component outputs
    components_dir = Path(config.FEATURES_OUTPUT_PATH) / "components"
    components_dir.mkdir(parents=True, exist_ok=True)
    
    base_path = components_dir / "openinsider_data.parquet"
    annual_path = components_dir / "all_annual_statements.parquet"
    tech_path = components_dir / "all_technical_indicators.parquet"
    macro_path = components_dir / "all_macro_data.parquet"

    # --- Step 1: Get base insider trading data ---
    print("--- Step 1: Scraping base insider data ---")
    # Scrape the requested number of weeks that all end at 6 months ago
    base_df = scrape_openinsider(num_weeks=num_weeks, start_months_ago=None, end_months_ago=6)
    if base_df.empty:
        print("No base data scraped. Halting.")
        return
    base_df["Filing Date"] = pd.to_datetime(base_df["Filing Date"])
    base_df.to_parquet(base_path, index=False)

    # --- Build OHLCV component parquets (past for TA, future for targets) ---
    print("--- Step 1b: Building OHLCV past/future components ---")
    build_event_ohlcv_datasets(
        base_df=base_df,
        db_path_str=str(config.STOOQ_DATABASE_PATH),
        past_output_path=Path(config.OHLCV_PAST_COMPONENT_PATH),
        future_output_path=Path(config.OHLCV_FUTURE_COMPONENT_PATH),
        past_lookback_calendar_days=400,
        future_lookahead_trading_days=126,
        n_jobs=-2,
    )

    # # --- Step 2, 3, 4: Generate feature components in parallel (conceptually) ---
    generate_annual_statements(base_df, annual_path, sec_parquet_dir=config.EDGAR_DOWNLOAD_PATH, request_header=config.REQUESTS_HEADER)
    # Technicals now read from ohlcv_past component to avoid lookahead
    generate_technical_indicators(base_df, config.STOOQ_DATABASE_PATH, tech_path)

    # --- Step 5: Merge all feature components ---
    print("\n--- Step 5: Merging all feature components ---")
    
    # Load primary components
    base_df = pd.read_parquet(base_path)
    annual_df = pd.read_parquet(annual_path)

    # Convert date columns for merging
    annual_df['Filing Date'] = pd.to_datetime(annual_df['Filing Date'])

    # Defensively handle empty annual statements
    if annual_df.empty:
        print("  [WARN] Annual statements file is empty. Proceeding without financial data.")
        merged_df = base_df.copy()
    else:
        # Start with an inner join to keep only tickers with financial data
        merged_df = pd.merge(base_df, annual_df, on=["Ticker", "Filing Date"], how="inner")
        print(f"  Rows after merging with annual statements: {len(merged_df)}")
    
    # --- THIS IS THE FIX: Defensively merge technical indicators ---
    if tech_path.exists():
        tech_df = pd.read_parquet(tech_path)
        if not tech_df.empty:
            merged_df = pd.merge(merged_df, tech_df, on=["Ticker", "Filing Date"], how="inner")
        else:
            print("  [WARN] Technical indicators file was created but is empty. Skipping merge.")
    else:
        print("  [WARN] Technical indicators file not found. Skipping merge.")
    # --- END OF FIX ---

    # --- Step 6: Save the final raw feature set ---
    raw_features_path = Path(config.FEATURES_OUTPUT_PATH) / "raw_features.parquet"
    merged_df.to_parquet(raw_features_path, index=False)
    
    end_time = time.time()
    print(f"\n✅ Feature scraping pipeline complete in {end_time - start_time:.2f} seconds.")
    print(f"Final raw feature set saved to {raw_features_path}")

