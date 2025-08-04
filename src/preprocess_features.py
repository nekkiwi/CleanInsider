# file: src/preprocess_features.py

import time
from pathlib import Path
import pandas as pd
from preprocess.time_based_splitter import create_time_based_splits
# --- THIS IS THE FIX ---
# Use a relative import (.) to tell Python to look in the current directory's
# 'preprocess' subfolder.
from .preprocess.fold_processor import FoldProcessor 
# --- END OF FIX ---
from preprocess.utils import save_columns_list

def run_preprocess_pipeline(config, n_splits=7, corr_thresh=0.8, var_thresh=0.0001, missing_thresh=0.6, start_date: str = None):
    """
    Main orchestrator for the walk-forward preprocessing pipeline.
    This robust version uses a stateful FoldProcessor to ensure consistency
    between training and evaluation sets, preventing KeyErrors.
    """
    print("--- Starting Walk-Forward Preprocessing Pipeline ---")
    start_time = time.time()

    # --- 1. Initial Setup and Global Filtering ---
    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    info_dir = Path(config.FEATURES_INFO_OUTPUT_PATH)
    input_path = features_dir / "raw_features.parquet"
    info_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows with {df.shape[1]} features.")

    if start_date:
        df['Filing Date'] = pd.to_datetime(df['Filing Date'])
        df = df[df['Filing Date'] >= pd.to_datetime(start_date)].copy()
        print(f"Filtered data to {len(df)} rows after start date.")

    # Apply global missingness filter
    missing_proportions = df.isnull().sum() / len(df)
    # Define protected features that should not be dropped
    PROTECTED_FEATURES = {
        'Ticker', 'Filing Date', 'Price', 'Value', 'Qty' 
    }
    cols_to_drop_missing = missing_proportions[missing_proportions > missing_thresh].index
    cols_to_drop_final = [col for col in cols_to_drop_missing if col not in PROTECTED_FEATURES]
    df.drop(columns=cols_to_drop_final, inplace=True, errors="ignore")
    print(f"Dropped {len(cols_to_drop_final)} features with >{missing_thresh*100}% missing values.")

    # --- 2. Create Time-Based Splits ---
    df_split, _ = create_time_based_splits(df, n_splits=n_splits)

    # --- 3. Walk-Forward Preprocessing and Saving (Single Pass) ---
    print("\n--- Processing folds to create Training and Evaluation sets ---")
    
    # We create n_splits-1 folds for walk-forward validation
    for i in range(1, n_splits):
        print(f"\n--- Generating data for Fold {i} ---")

        # Define the raw data for this fold's training and evaluation sets
        train_df_raw = df_split[df_split['split_id'] <= i].copy()
        eval_df_raw = df_split[df_split['split_id'] == i + 1].copy()
        
        print(f"  Training on {len(train_df_raw)} rows (Splits 1-{i}).")
        print(f"  Evaluating on {len(eval_df_raw)} rows (Split {i+1}).")

        # Initialize and fit the processor ONLY on the training data
        processor = FoldProcessor(corr_threshold=corr_thresh, variance_threshold=var_thresh)
        processor.fit(train_df_raw)

        # Transform both datasets using the SAME learned parameters
        train_df_processed = processor.transform(train_df_raw)
        eval_df_processed = processor.transform(eval_df_raw)
        
        # --- 4. Save the Final Datasets ---
        fold_output_dir = features_dir / f"fold_{i}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        train_save_path = fold_output_dir / "training_data.parquet"
        eval_save_path = fold_output_dir / "evaluation_data.parquet"
        
        train_df_processed.to_parquet(train_save_path, index=False)
        eval_df_processed.to_parquet(eval_save_path, index=False)
        
        print(f"✅ Fold {i} data saved:")
        print(f"   - Training set ({train_df_processed.shape[0]}x{train_df_processed.shape[1]}): {train_save_path}")
        print(f"   - Evaluation set ({eval_df_processed.shape[0]}x{eval_df_processed.shape[1]}): {eval_save_path}")
        
        # Save info for this fold
        fold_info_dir = info_dir / f"fold_{i}"
        fold_info_dir.mkdir(parents=True, exist_ok=True)
        save_columns_list(train_df_processed, fold_info_dir / "final_columns.txt")

    end_time = time.time()
    print(f"\n--- Preprocessing Complete in {end_time - start_time:.2f} seconds ---")
