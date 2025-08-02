# file: src/scrape_targets.py

import time
import pandas as pd
from pathlib import Path

# Import the new, low-level modules
from scrapers.target_scraper.data_loader import load_all_ohlcv_data
from scrapers.target_scraper.generate_targets import calculate_realized_alpha_series

def generate_targets_for_combinations(base_df: pd.DataFrame, db_path_str: str, target_combinations: list) -> pd.DataFrame:
    """
    Orchestrates the generation of multiple target columns.
    """
    unique_tickers = tuple(base_df['Ticker'].unique()) + ('^spx',)
    ohlcv_data = load_all_ohlcv_data(unique_tickers, db_path_str)
    spx_data = ohlcv_data.pop('^spx', pd.DataFrame())
    if spx_data.empty:
        raise ValueError("Could not load '^spx' data from the database.")

    results_df = base_df[['Ticker', 'Filing Date']].copy()
    
    for params in target_combinations:
        timepoint, tp, sl = params['time'], params['tp'], params['sl']
        col_name = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"
        print(f"\n--- Generating Target Column: {col_name} ---")
        
        alpha_series = calculate_realized_alpha_series(
            base_df=base_df, ohlcv_data=ohlcv_data, spx_data=spx_data,
            timepoint_str=timepoint, take_profit=tp, stop_loss=sl
        )
        results_df[col_name] = alpha_series
    
    return results_df

def run_target_generation_pipeline(config, target_combinations: list, n_splits: int = 7):
    """
    Main pipeline for generating multiple target combinations and saving them for each fold.
    """
    print("\n--- Starting Target Generation Pipeline (Multi-Combination) ---")
    start_time = time.time()

    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)
    targets_dir.mkdir(parents=True, exist_ok=True)

    print("Gathering all unique events to process...")
    largest_fold_path = features_dir / f"fold_{n_splits - 1}" / "preprocessed_fold.parquet"
    test_set_path = features_dir / "final_test_set_unprocessed.parquet"
    base_df = pd.concat([
        pd.read_parquet(largest_fold_path), pd.read_parquet(test_set_path)
    ])[['Ticker', 'Filing Date', 'Price']].drop_duplicates().reset_index(drop=True)

    master_targets_df = generate_targets_for_combinations(base_df, config.STOOQ_DATABASE_PATH, target_combinations)

    print("\n--- Merging all targets into each fold and saving final files ---")
    for i in range(1, n_splits):
        fold_features_dir = features_dir / f"fold_{i}"
        fold_targets_dir = targets_dir / f"fold_{i}"
        fold_targets_dir.mkdir(parents=True, exist_ok=True)
        
        df_fold_features = pd.read_parquet(fold_features_dir / "preprocessed_fold.parquet")
        df_fold_features['Filing Date'] = pd.to_datetime(df_fold_features['Filing Date'])

        df_fold_targets = pd.merge(df_fold_features[['Ticker', 'Filing Date']], master_targets_df, on=['Ticker', 'Filing Date'], how='inner')
        df_fold_targets.to_parquet(fold_targets_dir / "targets.parquet", index=False)
        print(f"✅ Saved multi-combination targets for Fold {i}.")

    df_test = pd.read_parquet(test_set_path)
    df_test['Filing Date'] = pd.to_datetime(df_test['Filing Date'])
    df_test_targets = pd.merge(df_test[['Ticker', 'Filing Date']], master_targets_df, on=['Ticker', 'Filing Date'], how='inner')
    df_test_targets.to_parquet(targets_dir / "test_set_targets.parquet", index=False)
    print("✅ Saved multi-combination targets for the test set.")

    end_time = time.time()
    print(f"\n--- Target Generation Complete in {end_time - start_time:.2f} seconds ---")
