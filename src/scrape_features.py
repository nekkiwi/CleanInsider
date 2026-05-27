# file: src/scrape_features.py

import time
from pathlib import Path

import pandas as pd

from src.scrapers.feature_scraper.load_annual_statements import (
    generate_annual_statements,
)
from src.scrapers.feature_scraper.load_macro_features import generate_macro_features
from src.scrapers.feature_scraper.load_technical_indicators import (
    generate_technical_indicators,
)
from src.scrapers.feature_scraper.prefetch_prices import prefetch_missing_tickers
from src.scrapers.feature_scraper.scrape_openinsider import scrape_openinsider


def run_feature_scraping_pipeline(num_weeks: int, config, rescrape: bool = True):
    """
    Orchestrates the component-based feature scraping pipeline.

    Args:
        num_weeks: Number of weeks of OpenInsider history to scrape.
        config: The project config module (paths, headers).
        rescrape: When True (default) re-scrape every component from source.
            When False, skip scraping and only re-merge the existing component
            parquets in data/scrapers/features/components/ (fast path).
    """
    start_time = time.time()

    # Define paths for component outputs
    components_dir = Path(config.FEATURES_OUTPUT_PATH) / "components"
    components_dir.mkdir(parents=True, exist_ok=True)

    base_path = components_dir / "openinsider_data.parquet"
    annual_path = components_dir / "all_annual_statements.parquet"
    tech_path = components_dir / "all_technical_indicators.parquet"
    macro_path = components_dir / "all_macro_data.parquet"

    if rescrape:
        # --- Step 1: Get base insider trading data ---
        print("--- Step 1: Scraping base insider data ---")
        base_df = scrape_openinsider(num_weeks=num_weeks)
        if base_df.empty:
            print("No base data scraped. Halting.")
            return
        base_df["Filing Date"] = pd.to_datetime(base_df["Filing Date"])
        base_df.to_parquet(base_path, index=False)

        # --- Recover delisted/absent tickers into the local Stooq cache ---
        # Batched yfinance pull for any event ticker missing locally, so the rest
        # of the pipeline can read prices local-only (fast) without losing the
        # survivorship-relevant delisted names.
        try:
            prefetch_missing_tickers(
                base_df["Ticker"].dropna().unique().tolist(),
                config.STOOQ_DATABASE_PATH,
            )
        except Exception as e:
            print(f"  [WARN] price prefetch failed (continuing): {e}")

        # --- Steps 2, 3, 4: Generate feature components ---
        print("--- Step 2: Generating annual statement (fundamental) features ---")
        generate_annual_statements(
            base_df,
            annual_path,
            sec_parquet_dir=config.EDGAR_DOWNLOAD_PATH,
            request_header=config.REQUESTS_HEADER,
        )
        print("--- Step 3: Generating technical indicators ---")
        generate_technical_indicators(base_df, config.STOOQ_DATABASE_PATH, tech_path)
        print("--- Step 4: Generating macro features ---")
        generate_macro_features(base_df, config.STOOQ_DATABASE_PATH, macro_path)
    else:
        print(
            "--- Steps 1-4 skipped (rescrape=False): re-merging existing components ---"
        )

    # --- Step 5: Merge all feature components ---
    print("\n--- Step 5: Merging all feature components ---")

    # Load primary components
    base_df = pd.read_parquet(base_path)
    annual_df = pd.read_parquet(annual_path)
    macro_df = pd.read_parquet(macro_path)

    # Convert date columns for merging
    annual_df["Filing Date"] = pd.to_datetime(annual_df["Filing Date"])
    macro_df["Filing Date"] = pd.to_datetime(macro_df["Filing Date"])

    # Defensively handle empty annual statements
    if annual_df.empty:
        print(
            "  [WARN] Annual statements file is empty. Proceeding without financial data."
        )
        merged_df = base_df.copy()
    else:
        # Start with an inner join to keep only tickers with financial data
        merged_df = pd.merge(
            base_df, annual_df, on=["Ticker", "Filing Date"], how="inner"
        )
        print(f"  Rows after merging with annual statements: {len(merged_df)}")

    # --- THIS IS THE FIX: Defensively merge technical indicators ---
    if tech_path.exists():
        tech_df = pd.read_parquet(tech_path)
        if not tech_df.empty:
            merged_df = pd.merge(
                merged_df, tech_df, on=["Ticker", "Filing Date"], how="inner"
            )
        else:
            print(
                "  [WARN] Technical indicators file was created but is empty. Skipping merge."
            )
    else:
        print("  [WARN] Technical indicators file not found. Skipping merge.")
    # --- END OF FIX ---

    merged_df = pd.merge(merged_df, macro_df, on="Filing Date", how="left")
    print(f"  Rows after merging all components: {len(merged_df)}")

    # --- Step 6: Save the final raw feature set ---
    raw_features_path = Path(config.FEATURES_OUTPUT_PATH) / "raw_features.parquet"
    merged_df.to_parquet(raw_features_path, index=False)

    end_time = time.time()
    print(
        f"\n✅ Feature scraping pipeline complete in {end_time - start_time:.2f} seconds."
    )
    print(f"Final raw feature set saved to {raw_features_path}")
