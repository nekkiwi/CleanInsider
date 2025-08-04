# file: src/preprocess/various_preprocessing.py

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize


def apply_additional_preprocessing(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Applies a series of additional, domain-specific preprocessing steps.
    Refactored to improve performance by reducing DataFrame fragmentation.
    """
    print("Applying additional preprocessing...")
    df = df.copy()  # Work on a copy to avoid SettingWithCopyWarning

    # --- 1. Outlier Handling ---
    print("   → Handling outliers with Winsorization...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_bounds = {}
    for col in numeric_cols:
        # Skip columns that are constant or near-constant
        if df[col].nunique() <= 1:
            continue
        lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
        if lower < upper:  # Ensure there is a range to clip
            outlier_bounds[col] = {"lower": lower, "upper": upper}
            df[col] = winsorize(df[col], limits=(0.01, 0.01))

    # --- 2. Feature Engineering ---
    # Create a dictionary to hold all new columns. This avoids fragmentation.
    new_cols = {}

    # Normalizing by Price
    print("   → Normalizing features by Price...")
    price_norm_candidates = [
        c for c in df.columns if "_q-" in c and "perShare" not in c
    ]
    for col in price_norm_candidates:
        if "Price" in df.columns and col in df.columns:
            new_col_name = f"{col}_per_Share"
            new_cols[new_col_name] = df[col] / df["Price"].replace(0, np.nan)

    # Normalizing by Market Cap
    print("   → Features normalized by Market Cap or Price where applicable.")
    if "CommonStockSharesOutstanding_q-1" in df.columns and "Price" in df.columns:
        market_cap = df["CommonStockSharesOutstanding_q-1"] * df["Price"]
        market_cap_norm_candidates = [
            "Assets_q-1",
            "Liabilities_q-1",
            "StockholdersEquity_q-1",
            "NetIncomeLoss_q-2",
            "StockholdersEquity_q-2",
        ]
        for col in market_cap_norm_candidates:
            if col in df.columns:
                new_col_name = f"{col}_per_MarketCap"
                new_cols[new_col_name] = df[col] / market_cap.replace(0, np.nan)

    # Aggregate and ratio features
    print("   → Aggregate and ratio features added.")
    if "Assets_q-1" in df.columns and "Liabilities_q-1" in df.columns:
        new_cols["DebtToAssets_q-1"] = df["Liabilities_q-1"] / df["Assets_q-1"].replace(
            0, np.nan
        )
    if "NetIncomeLoss_q-2" in df.columns and "StockholdersEquity_q-2" in df.columns:
        new_cols["ROE_q-2"] = df["NetIncomeLoss_q-2"] / df[
            "StockholdersEquity_q-2"
        ].replace(0, np.nan)

    # Log transforms for skewed data
    print("   → Log transforms added.")
    log_candidates = ["Price", "Value", "Qty"]
    for col in log_candidates:
        if col in df.columns:
            new_col_name = f"log_{col}"
            # Add a small constant to handle zeros before log transform
            new_cols[new_col_name] = np.log(df[col] + 1)

    # --- Combine all new columns at once ---
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Replace any new inf/-inf values created during division
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df, outlier_bounds
