# Use your existing pruning functions (untouched)
import json
import time
from pathlib import Path

import pandas as pd

from preprocess.prune_features import (
    parse_missing_data_report,
    prune_feature_set,
    save_columns,
)
from preprocess.various_preprocessing import apply_additional_preprocessing


def run_preprocess_pipeline(
    config, missing_thresh=70.0, corr_thresh=0.8, var_thresh=0.01
):
    print("--- Starting Preprocessing Pipeline ---")
    start_time = time.time()

    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    info_dir = Path(config.FEATURES_INFO_OUTPUT_PATH)
    input_path = features_dir / "raw_features.parquet"
    missing_report_path = info_dir / "missing_data_report.txt"
    output_path = features_dir / "preprocessed_features.parquet"
    outlier_bounds_path = info_dir / "outlier_clip_bounds.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Merged features file not found at {input_path}. Run scraping pipeline first."
        )
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows with {df.shape[1]} features from {input_path}.")

    # --- Integrated Pruning Logic (Based on run_prune_pipeline) ---
    print("\n--- Step 1: Pruning features ---")
    missing_data_map = parse_missing_data_report(missing_report_path)
    pruned_df = prune_feature_set(
        df=df,
        missing_data=missing_data_map,
        missing_threshold=missing_thresh,
        corr_threshold=corr_thresh,
        variance_threshold=var_thresh,
    )

    # --- Final Step: Save ONLY the pruned columns list ---
    print("\n--- Saving pruned columns list ---")
    save_columns(pruned_df, output_txt_path=info_dir / "pruned_features_columns.txt")

    # --- Step 2: Additional Preprocessing ---
    print("\n--- Step 2: Applying additional preprocessing ---")
    preprocessed_df, outlier_bounds = apply_additional_preprocessing(pruned_df)

    # Save the final preprocessed dataset
    preprocessed_df.to_parquet(output_path, index=False)
    print(f"✅ Final preprocessed dataset saved to {output_path}.")

    # Save the outlier clipping bounds for reproducibility
    with open(outlier_bounds_path, "w") as f:
        json.dump(outlier_bounds, f, indent=2)
    print(f"✅ Outlier clipping bounds saved to {outlier_bounds_path}.")

    # --- Final Step: Save ONLY the pruned columns list ---
    print("\n--- Saving preprocessed columns list ---")
    save_columns(
        preprocessed_df, output_txt_path=info_dir / "preprocessed_features_columns.txt"
    )

    end_time = time.time()
    print(
        f"\n--- Preprocessing Pipeline Complete in {end_time - start_time:.2f} seconds ---"
    )
