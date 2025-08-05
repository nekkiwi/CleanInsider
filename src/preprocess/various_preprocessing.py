# file: src/preprocess/various_preprocessing.py

import numpy as np
import pandas as pd


def apply_additional_preprocessing(
    df: pd.DataFrame, existing_bounds: dict = None
) -> (pd.DataFrame, dict):
    """
    Applies a series of additional, domain-specific preprocessing steps adapted
    for annual yfinance data.
    """
    print("Applying additional preprocessing...")
    df = df.copy()

    # --- 1. Outlier Handling (Remains the same) ---
    print(" → Handling outliers...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].astype("float64")
    newly_calculated_bounds = {}
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            continue
        if existing_bounds and col in existing_bounds:
            lower, upper = existing_bounds[col]["lower"], existing_bounds[col]["upper"]
        else:
            lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
            if lower < upper:
                newly_calculated_bounds[col] = {"lower": lower, "upper": upper}
        if lower < upper:
            df[col] = df[col].clip(lower, upper)
    outlier_bounds_to_return = (
        existing_bounds if existing_bounds is not None else newly_calculated_bounds
    )

    # --- 2. Feature Engineering (ADAPTED FOR ANNUAL DATA) ---
    print(" → Engineering new features from annual data...")
    new_cols = {}

    # Define key annual metrics based on yfinance column names
    # Using .get() provides a safe way to access columns that might not exist for all stocks
    total_assets = df.get("FIN_Total Assets_Y1")
    total_liabilities = df.get("FIN_Total Liabilities Net Minority Interest_Y1")
    net_income = df.get("FIN_Net Income_Y1")
    stockholders_equity = df.get("FIN_Stockholders Equity_Y1")
    shares_outstanding = df.get(
        "FIN_Share Issued_Y1"
    )  # Using 'Share Issued' as proxy for outstanding

    # --- Ratios based on the most recent year (Y1) ---
    if total_assets is not None and total_liabilities is not None:
        new_cols["FE_DebtToAssets_Y1"] = total_liabilities / total_assets.replace(
            0, np.nan
        )

    if net_income is not None and stockholders_equity is not None:
        new_cols["FE_ROE_Y1"] = net_income / stockholders_equity.replace(0, np.nan)

    # --- Per-share metrics ---
    if shares_outstanding is not None:
        if net_income is not None:
            new_cols["FE_EPS_Y1"] = net_income / shares_outstanding.replace(0, np.nan)
        if total_assets is not None:
            new_cols["FE_Assets_per_Share_Y1"] = (
                total_assets / shares_outstanding.replace(0, np.nan)
            )

    # --- Log transforms for skewed data (remains the same) ---
    log_candidates = ["Price", "Value", "Qty"]
    for col in log_candidates:
        if col in df.columns:
            new_cols[f"log_{col}"] = np.log(df[col] + 1)

    # Combine all new columns at once
    if new_cols:
        new_cols_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_cols_df], axis=1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df, outlier_bounds_to_return
