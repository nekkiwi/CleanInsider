# file: scrape_data.py

import argparse

from src import config  # Import the central config
from src.preprocess_features import run_preprocess_pipeline
from src.scrape_features import run_feature_scraping_pipeline
from src.scrape_targets import run_target_generation_pipeline
from src.scrapers.feature_scraper.feature_scraper_util.general_utils import (
    create_output_directories,
)


def main(num_weeks: int):
    """
    Main entry point for the data scraping pipeline.
    """
    print("--- Starting Data Scraping Pipeline ---")

    # 1. Setup environment using paths from the config file
    print("Step 1: Setting up environment and creating directories...")
    create_output_directories(config.DIRECTORIES_TO_CREATE)
    print(f"Data will be saved in: {config.FEATURES_OUTPUT_PATH}")
    print("...Environment setup complete.\n")

    # 2. Run the full feature scraping pipeline
    n_splits = 7
    run_feature_scraping_pipeline(num_weeks=num_weeks, config=config)

    # 3. Run the preprocessing pipeline
    # The start_date argument has been removed as it is no longer needed
    # by the new version of the preprocessing script.
    run_preprocess_pipeline(
        config=config,
        num_folds=5,  # This will generate 5 validation folds and 1 test set
        corr_thresh=0.8,
        var_thresh=0.0001,
        missing_thresh=0.6,
    )

    # --- Target Generation Pipeline (can be run after preprocessing) ---
    target_combinations = [
        {"time": "1w", "tp": 0.05, "sl": -0.05},
        # Add other combinations as needed
    ]

    # Uncomment the following lines to run target generation
    run_target_generation_pipeline(
        config=config,
        target_combinations=target_combinations,
        n_splits=n_splits,
        batch_size=100,
        debug=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full data scraping pipeline.")
    parser.add_argument(
        "--weeks",
        type=int,
        default=3,
        help="Number of weeks of insider trading data to scrape. Default is 3.",
    )
    args = parser.parse_args()
    main(num_weeks=args.weeks)
