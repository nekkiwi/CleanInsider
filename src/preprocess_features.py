# file: src/preprocess_features.py

import time
from pathlib import Path

import pandas as pd

from .preprocess.fold_processor import FoldProcessor
from .preprocess.time_based_splitter import create_time_based_splits
from .preprocess.utils import save_columns_list
from .scrapers.feature_scraper.feature_scraper_util.general_utils import (
    report_missing_data,
)


def run_preprocess_pipeline(
    config, num_folds=5, corr_thresh=0.8, var_thresh=0.0001, missing_thresh=0.6
):
    """
    Orchestrates a robust, two-pass walk-forward pipeline.
    It generates 'num_folds' for validation and one final, held-out test set,
    ensuring no data leakage.
    """
    print("\n--- Starting 2-Pass Preprocessing Pipeline with Final Test Set ---")
    start_time = time.time()

    # --- 0. Initial Setup ---
    # With num_folds for validation and one for test, we need more splits total.
    n_total_splits = num_folds + 2

    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    info_dir = Path(config.FEATURES_INFO_OUTPUT_PATH) / "preprocessing"
    info_dir.mkdir(parents=True, exist_ok=True)
    input_path = features_dir / "raw_features.parquet"

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows with {df.shape[1]} features.")
    df_split, _ = create_time_based_splits(df, n_splits=n_total_splits)
    print(f"Data divided into {n_total_splits} time-based splits.")

    # --- PASS 1: Learn processing parameters from each validation fold's training data ---
    print("\n--- PASS 1: Learning features from each training fold ---")
    fold_processors = {}
    fold_surviving_features = []

    for i in range(1, num_folds + 1):
        print(f"\n  > Learning from Fold {i} (uses Splits 1-{i} for training)")
        train_df_raw = df_split[df_split["split_id"] <= i].copy()

        report_missing_data(train_df_raw, output_dir=info_dir / f"fold_{i}")

        processor = FoldProcessor(
            corr_threshold=corr_thresh, variance_threshold=var_thresh
        )
        processor.fit(train_df_raw)
        fold_processors[i] = processor

        processed_df = processor.transform(train_df_raw)
        fold_surviving_features.append(set(processed_df.columns))

    # --- Determine the common features across all validation training folds ---
    if not fold_surviving_features:
        print("❌ CRITICAL: No features survived Pass 1. Halting.")
        return

    common_features = sorted(list(set.intersection(*fold_surviving_features)))
    save_columns_list(
        pd.DataFrame(columns=common_features), info_dir / "common_features.txt"
    )
    print(
        f"\n--- Found {len(common_features)} features common across all {num_folds} training folds. ---"
    )

    # --- PASS 2: Apply transformations and save final datasets ---
    print("\n--- PASS 2: Applying transformations and saving final fold data ---")

    # Process and save the validation folds (1 to num_folds)
    for i in range(1, num_folds + 1):
        print(f"  > Processing and saving Validation Fold {i}")
        fold_output_dir = features_dir / f"fold_{i}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        processor = fold_processors[i]

        train_df_raw = df_split[df_split["split_id"] <= i].copy()
        eval_df_raw = df_split[df_split["split_id"] == i + 1].copy()

        train_df_processed = processor.transform(train_df_raw)
        eval_df_processed = processor.transform(eval_df_raw)

        # Filter to common features
        train_df_final = train_df_processed[
            [col for col in common_features if col in train_df_processed.columns]
        ]
        eval_df_final = eval_df_processed[
            [col for col in common_features if col in eval_df_processed.columns]
        ]

        train_df_final.to_parquet(
            fold_output_dir / "training_data.parquet", index=False
        )
        eval_df_final.to_parquet(
            fold_output_dir / "validation_data.parquet", index=False
        )
        print(
            f"  ✅ Fold {i} saved with {len(train_df_final.columns)} common features."
        )

    # --- Final Test Set Processing ---
    print("\n  > Processing and saving the Final Test Set")
    test_set_dir = features_dir / "test_set"
    test_set_dir.mkdir(parents=True, exist_ok=True)

    # The test set is the last split
    test_df_raw = df_split[df_split["split_id"] == n_total_splits].copy()

    # Use the processor from the LARGEST training set (from the last validation fold)
    # This is the most robust choice and prevents any leakage from the test set itself.
    final_processor = fold_processors[num_folds]

    test_df_processed = final_processor.transform(test_df_raw)

    # Filter to the same common feature set
    test_df_final = test_df_processed[
        [col for col in common_features if col in test_df_processed.columns]
    ]

    test_df_final.to_parquet(test_set_dir / "test_data.parquet", index=False)
    print(
        f"  ✅ Final test set saved with {len(test_df_final.columns)} common features."
    )

    end_time = time.time()
    print(f"\n--- Preprocessing Complete in {end_time - start_time:.2f} seconds ---")
