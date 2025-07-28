import argparse

from src import config  # Import the central config
from src.scrape_features import run_feature_scraping_pipeline
from src.scrapers.scrape_feature_utils.general_utils import create_output_directories


def main(num_weeks: int):
    """
    Main entry point for the data scraping pipeline.
    """
    print("--- Starting Data Scraping Pipeline ---")

    # 1. Setup environment using paths from the config file
    print("Step 1: Setting up environment and creating directories...")
    create_output_directories(config.DIRECTORIES_TO_CREATE)
    print(f"Data will be saved in: {config.FEATURES_DIR}")
    print("...Environment setup complete.\n")

    # 2. Run the full feature scraping pipeline
    run_feature_scraping_pipeline(
        num_weeks=num_weeks,
        config=config  # Pass the entire config object
    )

    print("--- Data Scraping Pipeline Finished Successfully ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full data scraping pipeline.")
    parser.add_argument(
        "--weeks",
        type=int,
        default=52,
        help="Number of weeks of insider trading data to scrape. Default is 52."
    )
    args = parser.parse_args()
    main(num_weeks=args.weeks)
