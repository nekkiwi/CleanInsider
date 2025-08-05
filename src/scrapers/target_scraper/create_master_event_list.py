# file: src/scrapers/target_scraper/create_master_event_list.py

import pandas as pd
from pathlib import Path

def create_master_event_list(config, n_splits: int):
    """
    Gathers all unique (Ticker, Filing Date, Price) events that need a target calculated.
    This reads from the final, processed training, validation, and test sets.
    """
    print("\n--- STEP 1: Creating Master Event List ---")
    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)
    targets_dir.mkdir(parents=True, exist_ok=True)
    
    print("Gathering all unique events from all processed data files...")
    
    all_events = []
    required_cols = ['Ticker', 'Filing Date', 'Price']
    
    # Gather events from all training and validation folds
    for i in range(1, n_splits + 1):
        train_path = features_dir / f"fold_{i}" / "training_data.parquet"
        val_path = features_dir / f"fold_{i}" / "validation_data.parquet"
        
        if train_path.exists():
            df_train = pd.read_parquet(train_path)
            if all(col in df_train.columns for col in required_cols):
                all_events.append(df_train[required_cols])
        
        if val_path.exists():
            df_val = pd.read_parquet(val_path)
            if all(col in df_val.columns for col in required_cols):
                all_events.append(df_val[required_cols])
                
    # Add events from the final test set
    test_path = features_dir / "test_set" / "test_data.parquet"
    if test_path.exists():
        df_test = pd.read_parquet(test_path)
        if all(col in df_test.columns for col in required_cols):
            all_events.append(df_test[required_cols])

    if not all_events:
        raise FileNotFoundError("No feature files found. Please run the feature preprocessing pipeline first.")

    base_df = pd.concat(all_events).drop_duplicates(subset=['Ticker', 'Filing Date'])
    
    # --- Data validation and cleaning ---
    original_count = len(base_df)
    base_df.dropna(subset=['Price'], inplace=True)
    base_df = base_df[base_df['Price'] > 0].copy()
    cleaned_count = len(base_df)
    
    if original_count > cleaned_count:
        print(f" -> Cleaned event list: Removed {original_count - cleaned_count} rows with invalid prices (NaN or <= 0).")
        
    base_df = base_df.sort_values(by=['Ticker', 'Filing Date']).reset_index(drop=True)
    
    output_path = targets_dir / "master_event_list.parquet"
    base_df.to_parquet(output_path, index=False)
    
    print(f"✅ Found {len(base_df)} unique events. Master list saved to {output_path}")

