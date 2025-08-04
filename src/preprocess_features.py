# file: src/preprocess_features.py

import json
import time
from pathlib import Path

import pandas as pd

from preprocess.fold_processor import process_fold_data
from preprocess.time_based_splitter import create_time_based_splits
from preprocess.utils import save_columns_list


def run_preprocess_pipeline(
    config,
    n_splits=7,
    corr_thresh=0.8,
    var_thresh=0.0001,
    missing_thresh=0.6,
    start_date: str = None,
):
    """
    Main orchestrator for the walk-forward preprocessing pipeline.
    """
    print("--- Starting Walk-Forward Preprocessing Pipeline ---")
    start_time = time.time()

    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    info_dir = Path(config.FEATURES_INFO_OUTPUT_PATH)
    input_path = features_dir / "raw_features.parquet"

    info_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows with {df.shape[1]} features from {input_path}.")

    if start_date:
        print(
            f"\n--- Applying Start Date Filter: Keeping data on or after {start_date} ---"
        )
        original_rows = len(df)
        df["Filing Date"] = pd.to_datetime(df["Filing Date"])
        df = df[df["Filing Date"] >= pd.to_datetime(start_date)].copy()
        print(f"✅ Filtered data from {original_rows} to {len(df)} rows.")
        if len(df) == 0:
            raise ValueError("No data remains after applying the start date filter.")

    print(
        f"\n--- Applying Global Missingness Filter (> {missing_thresh*100}% missing) ---"
    )
    PROTECTED_FEATURES = {
        "Ticker",
        "Filing Date",
        "Price",
        "CommonStockSharesOutstanding_q-1",
        "Value",
        "Qty",
        "Pres_Buy_Value",
        "CFO_Buy_Value",
        "Assets_q-1",
        "Liabilities_q-1",
        "StockholdersEquity_q-1",
        "NetIncomeLoss_q-2",
        "StockholdersEquity_q-2",
        "CashAndCashEquivalentsAtCarryingValue_q-1",
        "NetCashProvidedByUsedInFinancingActivities_q-1",
        "NetCashProvidedByUsedInInvestingActivities_q-1",
        "Market_SPX_Volume",
        "Market_VIXY_Volume",
    }
    missing_proportions = df.isnull().sum() / len(df)
    cols_to_drop_missing = set(
        missing_proportions[missing_proportions > missing_thresh].index
    )
    cols_to_drop_missing = [
        col for col in cols_to_drop_missing if col not in PROTECTED_FEATURES
    ]
    df.drop(columns=cols_to_drop_missing, inplace=True, errors="ignore")
    print(
        f"✅ Dropped {len(cols_to_drop_missing)} features. {df.shape[1]} features remain."
    )

    save_columns_list(df, info_dir / "global_missingness_pruned_features.txt")

    df_split, ticker_date_map = create_time_based_splits(df, n_splits=n_splits)
    with open(info_dir / "time_split_ticker_map.json", "w") as f:
        json.dump(ticker_date_map, f, indent=4)
    print("✅ Ticker-FilingDate split map saved.")

    print("\n--- PASS 1: Initial processing of each fold (internal pruning) ---")
    fold_column_sets = []
    for i in range(1, n_splits):
        print(f"\n--- Processing Fold {i} (Splits 1-{i}) ---")
        fold_data = df_split[df_split["split_id"] <= i].copy()
        fold_preprocessed_df, fold_outlier_bounds = process_fold_data(
            fold_data, corr_threshold=corr_thresh, variance_threshold=var_thresh
        )
        fold_info_dir = info_dir / f"fold_{i}"
        fold_info_dir.mkdir(parents=True, exist_ok=True)

        temp_output_path = features_dir / f"temp_preprocessed_fold_{i}.parquet"
        fold_preprocessed_df.to_parquet(temp_output_path, index=False)

        with open(fold_info_dir / "outlier_clip_bounds.json", "w") as f:
            json.dump(fold_outlier_bounds, f, indent=4)

        fold_column_sets.append(set(fold_preprocessed_df.columns))
        print(f"✅ Fold {i} initial processing complete.")

    common_features = sorted(list(set.intersection(*fold_column_sets)))
    common_features_path = info_dir / "common_features_across_folds.txt"
    save_columns_list(pd.DataFrame(columns=common_features), common_features_path)
    print(f"\n--- Found {len(common_features)} common features across all folds. ---")

    print("\n--- PASS 2: Filtering folds to common feature set and finalizing ---")
    for i in range(1, n_splits):
        # Path for info files (.json, .txt) - remains in the info directory
        fold_info_dir = info_dir / f"fold_{i}"

        # --- NEW: Path for the preprocessed .parquet file ---
        fold_features_dir = features_dir / f"fold_{i}"
        fold_features_dir.mkdir(parents=True, exist_ok=True)

        temp_output_path = features_dir / f"temp_preprocessed_fold_{i}.parquet"
        df_fold = pd.read_parquet(temp_output_path)
        df_filtered = df_fold[common_features]

        # --- CHANGED: Save final parquet to the new features/fold_i directory ---
        final_output_path = fold_features_dir / "preprocessed_fold.parquet"
        df_filtered.to_parquet(final_output_path, index=False)

        # Save the final column list to the info/fold_i directory (this path is unchanged)
        cols_path = fold_info_dir / "preprocessed_columns.txt"
        save_columns_list(df_filtered, cols_path)

        temp_output_path.unlink()
        print(f"✅ Final data for Fold {i} saved to {final_output_path}")

    test_set = df_split[df_split["split_id"] == n_splits].copy()
    test_set_path = features_dir / "final_test_set_unprocessed.parquet"
    test_set.to_parquet(test_set_path, index=False)
    print(f"\n✅ Final test set saved to {test_set_path}")

    end_time = time.time()
    print(
        f"\n--- Walk-Forward Preprocessing Complete in {end_time - start_time:.2f} seconds ---"
    )
