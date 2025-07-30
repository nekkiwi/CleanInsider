# file: /src/prune_features.py

from pathlib import Path

import numpy as np
import pandas as pd


def parse_missing_data_report(filepath: Path) -> dict:
    """
    Parses the missing_data_report.txt file into a dictionary.

    Args:
        filepath (Path): The path to the missing data report.

    Returns:
        dict: A dictionary mapping feature names to their missing percentage.
    """
    if not filepath.exists():
        print(
            f"Warning: Missing data report not found at {filepath}. Cannot filter by missing %."
        )
        return {}

    missing_data = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # Skip headers or empty lines
            if (
                not line
                or line.startswith("Percentage")
                or line.startswith("No missing")
            ):
                continue

            # The feature name can have spaces, so we split from the right
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue

            feature, missing_pct_str = parts
            try:
                missing_data[feature] = float(missing_pct_str)
            except ValueError:
                continue  # Skip lines that can't be parsed

    return missing_data


def save_columns(pruned_df: pd.DataFrame, output_txt_path: Path) -> None:
    """
    Save the pruned DataFrame's column names to a text file in the features directory.
    """
    # Ensure the output directory exists
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # Write out sorted column names
    cols = sorted(pruned_df.columns.tolist())
    with open(output_txt_path, "w") as f:
        for col in cols:
            f.write(f"{col}\n")
    print(f"✅ List of pruned features saved to: {output_txt_path}")


def prune_feature_set(
    df: pd.DataFrame,
    missing_data: dict,
    missing_threshold: float,
    corr_threshold: float,
    variance_threshold: float,
) -> pd.DataFrame:
    original_feature_count = df.shape[1]

    # --- Step 1: Filter by Missing Data (Unchanged) ---
    features_to_drop_missing = {
        feature for feature, pct in missing_data.items() if pct > missing_threshold
    }
    df.drop(columns=list(features_to_drop_missing), inplace=True, errors="ignore")
    print(
        f"1. Missing Data Filter: Dropped {len(features_to_drop_missing)} features with >{missing_threshold}% missing values."
    )

    # --- Step 2: Intelligently Filter by Correlation (Unchanged) ---
    # pandas.corr() handles NaNs gracefully by default, so no changes are needed here.
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    features_to_drop_corr = set()
    while True:
        highly_corr_pairs = (upper_triangle > corr_threshold).stack()
        if not highly_corr_pairs.any():
            break
        feat1, feat2 = highly_corr_pairs.idxmax()
        mean_corr_feat1 = corr_matrix[feat1].mean()
        mean_corr_feat2 = corr_matrix[feat2].mean()
        next_to_drop = feat1 if mean_corr_feat1 > mean_corr_feat2 else feat2
        features_to_drop_corr.add(next_to_drop)
        upper_triangle.drop(columns=[next_to_drop], inplace=True, errors="ignore")
        upper_triangle.drop(index=[next_to_drop], inplace=True, errors="ignore")
    df.drop(columns=list(features_to_drop_corr), inplace=True, errors="ignore")
    print(
        f"2. Correlation Filter: Dropped {len(features_to_drop_corr)} features with correlation >{corr_threshold}."
    )

    # --- Step 3: Sanitize Infinity Values ---
    # We still need to handle infinities, but we will turn them into NaNs, not fill them.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("\nSanitizing data: Replaced all infinity values with NaN.")

    # --- STEP 4: REWRITTEN VARIANCE FILTER (COLUMN-BY-COLUMN) ---
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        features_to_drop_variance = set()
        for col in numeric_cols:
            # For each column, calculate its variance. The .var() method
            # automatically handles NaNs by ignoring them.
            variance = df[col].var()

            # Mark for dropping if variance is NaN (e.g., only one data point) or below the threshold
            if pd.isna(variance) or variance <= variance_threshold:
                features_to_drop_variance.add(col)

        # Drop the identified low-variance columns
        df.drop(columns=list(features_to_drop_variance), inplace=True, errors="ignore")
        dropped_variance_count = len(features_to_drop_variance)
    else:
        dropped_variance_count = 0

    print(
        f"3. Variance Filter: Dropped {dropped_variance_count} features with variance <={variance_threshold}."
    )

    final_feature_count = df.shape[1]
    print(
        f"\nPruning complete. Original features: {original_feature_count}, Final features: {final_feature_count}"
    )
    return df
