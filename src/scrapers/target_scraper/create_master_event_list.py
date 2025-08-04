# file: src/scrapers/target_scraper/1_create_master_event_list.py
import pandas as pd
from pathlib import Path

def create_master_event_list(config, n_splits: int):
    """
    Gathers all unique (Ticker, Filing Date, Price) events that need a target calculated.
    This is the first, fast step in a multi-step pipeline.
    """
    print("\n--- STEP 1: Creating Master Event List ---")
    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)
    targets_dir.mkdir(parents=True, exist_ok=True)

    print("Gathering all unique events from feature files...")
    largest_fold_path = features_dir / f"fold_{n_splits - 1}" / "preprocessed_fold.parquet"
    test_set_path = features_dir / "final_test_set_unprocessed.parquet"

    if not largest_fold_path.exists() or not test_set_path.exists():
        raise FileNotFoundError("Feature files for folds or test set not found. Please run the preprocessing pipeline first.")

    df_largest_fold = pd.read_parquet(largest_fold_path)
    df_test = pd.read_parquet(test_set_path)
    
    # Ensure 'Price' column is present for the target calculation
    required_cols = ['Ticker', 'Filing Date', 'Price']
    if not all(col in df_largest_fold.columns for col in required_cols) or \
       not all(col in df_test.columns for col in required_cols):
        raise KeyError("One of the input files is missing the required 'Ticker', 'Filing Date', or 'Price' columns.")

    base_df = pd.concat([
        df_largest_fold[required_cols], 
        df_test[required_cols]
    ]).drop_duplicates()
    
    # --- NEW: Add data validation and cleaning ---
    original_count = len(base_df)
    # 1. Drop rows where 'Price' is missing (NaN)
    base_df.dropna(subset=['Price'], inplace=True)
    # 2. Drop rows where 'Price' is not positive
    base_df = base_df[base_df['Price'] > 0].copy()
    
    cleaned_count = len(base_df)
    if original_count > cleaned_count:
        print(f"   -> Cleaned event list: Removed {original_count - cleaned_count} rows with invalid prices (NaN or <= 0).")
    # --- END OF NEW CODE ---

    base_df = base_df.sort_values(by=['Ticker', 'Filing Date']).reset_index(drop=True)
    
    cleaned_count = len(base_df)
    if original_count > cleaned_count:
        print(f"   -> Cleaned event list: Removed {original_count - cleaned_count} rows with invalid prices (NaN or <= 0).")
    
    output_path = targets_dir / "master_event_list.parquet"
    base_df.to_parquet(output_path, index=False)
    
    print(f"✅ Found {len(base_df)} unique events. Master list saved to {output_path}")

if __name__ == '__main__':
    # This allows running the step independently
    from src import config
    create_master_event_list(config, n_splits=7)
