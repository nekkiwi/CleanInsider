# file: src/preprocess/various_preprocessing.py

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def apply_additional_preprocessing(df: pd.DataFrame, existing_bounds: dict = None) -> (pd.DataFrame, dict):
    """
    Applies a series of additional, domain-specific preprocessing steps.
    If existing_bounds is provided, it uses them for Winsorization. Otherwise, it calculates them.
    """
    print("Applying additional preprocessing...")
    df = df.copy() 

    # --- 1. Outlier Handling ---
    print(" → Handling outliers with Winsorization...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].astype('float64')
    newly_calculated_bounds = {}

    for col in numeric_cols:
        if df[col].nunique() <= 1: continue

        # --- THIS IS THE FIX ---
        if existing_bounds and col in existing_bounds:
            # Use the pre-calculated bounds from the training set
            lower, upper = existing_bounds[col]['lower'], existing_bounds[col]['upper']
        else:
            # Calculate bounds if they don't exist (for the fit step)
            lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
            if lower < upper:
                newly_calculated_bounds[col] = {'lower': lower, 'upper': upper}
        
        # Clip the data using the determined bounds
        if lower < upper:
            df[col] = df[col].clip(lower, upper)
            
    # Determine which bounds to return
    outlier_bounds_to_return = existing_bounds if existing_bounds is not None else newly_calculated_bounds

    # --- 2. Feature Engineering (Unchanged) ---
    new_cols = {}
    price_norm_candidates = [c for c in df.columns if "_q-" in c and "perShare" not in c]
    for col in price_norm_candidates:
        if 'Price' in df.columns and col in df.columns:
            new_cols[f"{col}_per_Share"] = df[col] / df["Price"].replace(0, np.nan)

    if "CommonStockSharesOutstanding_q-1" in df.columns and "Price" in df.columns:
        market_cap = df["CommonStockSharesOutstanding_q-1"] * df["Price"]
        market_cap_norm_candidates = ['Assets_q-1', 'Liabilities_q-1', 'StockholdersEquity_q-1', 'NetIncomeLoss_q-2', 'StockholdersEquity_q-2']
        for col in market_cap_norm_candidates:
            if col in df.columns:
                new_cols[f"{col}_per_MarketCap"] = df[col] / market_cap.replace(0, np.nan)

    if 'Assets_q-1' in df.columns and 'Liabilities_q-1' in df.columns:
        new_cols['DebtToAssets_q-1'] = df['Liabilities_q-1'] / df['Assets_q-1'].replace(0, np.nan)
    if 'NetIncomeLoss_q-2' in df.columns and 'StockholdersEquity_q-2' in df.columns:
        new_cols['ROE_q-2'] = df['NetIncomeLoss_q-2'] / df['StockholdersEquity_q-2'].replace(0, np.nan)

    log_candidates = ['Price', 'Value', 'Qty']
    for col in log_candidates:
        if col in df.columns:
            new_cols[f"log_{col}"] = np.log(df[col] + 1)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df, outlier_bounds_to_return
