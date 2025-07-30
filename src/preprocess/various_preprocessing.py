# file: src/preprocess_features.py

import pandas as pd
import numpy as np

def apply_additional_preprocessing(df: pd.DataFrame):
    print("Applying additional preprocessing...")
    
    # Step 1: Handle outliers with Winsorization (clipping)
    print("   → Handling outliers with Winsorization...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_bounds = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.01)
        q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q3)
        outlier_bounds[col] = {'lower': q1, 'upper': q3}
    
    # ---- Normalization by Market Cap or Price ----
    # Step 1: build market cap proxy if not already present and if required columns exist
    # Try to use "CommonStockSharesOutstanding_q-1" and "Price"
    # Step 2: Price-normalize specific columns
    print("   → Normalizing features by Price...")
    has_market_cap_proxy = False
    if 'CommonStockSharesOutstanding_q-1' in df.columns and 'Price' in df.columns:
        df['Market_Cap_Proxy'] = df['CommonStockSharesOutstanding_q-1'] * df['Price']
        has_market_cap_proxy = True
    
    # Normalization targets: scale monetary amounts and volumes by Market Cap or Price
    norm_marketcap_candidates = [
        'Assets_q-1', 'Liabilities_q-1', 'StockholdersEquity_q-1', 'RetainedEarningsAccumulatedDeficit_q-1',
        'PropertyPlantAndEquipmentNet_q-1', 'NetIncomeLoss_q-2',
        'NetCashProvidedByUsedInFinancingActivities_q-1',
        'NetCashProvidedByUsedInInvestingActivities_q-1',
        'NetCashProvidedByUsedInOperatingActivities_q-1',
        'Value', 'Pres_Buy_Value', 'CFO_Buy_Value',
    ]
    for col in norm_marketcap_candidates:
        col_cap = f"{col}_per_MarketCap"
        if col in df.columns and has_market_cap_proxy:
            df[col_cap] = df[col] / df['Market_Cap_Proxy'].replace(0, np.nan)
    
    norm_price_candidates = [
        'Qty', 'Value', 'Pres_Buy_Value', 'CFO_Buy_Value'
    ]
    normalized_series_list = []
    for col in norm_price_candidates:
        if col in df.columns:
            col_norm_price = f"{col}_norm_price"
            normalized_series = df[col] / df['Price'].replace(0, np.nan)
            normalized_series.name = col_norm_price  # Set the name for the new column
            normalized_series_list.append(normalized_series)
    
    # Add all new columns at once using pd.concat
    if normalized_series_list:
        df = pd.concat([df] + normalized_series_list, axis=1)
    print("   → Features normalized by Market Cap or Price where applicable.")
    
    # ---- Logical Ratios & Aggregates ----
    # Debt-to-Equity
    if all(col in df.columns for col in ['Liabilities_q-1', 'StockholdersEquity_q-1']):
        df['Debt_to_Equity'] = df['Liabilities_q-1'] / df['StockholdersEquity_q-1'].replace(0, np.nan)
    # Cash Ratio
    if all(col in df.columns for col in ['CashAndCashEquivalentsAtCarryingValue_q-1', 'Liabilities_q-1']):
        df['Cash_Ratio'] = df['CashAndCashEquivalentsAtCarryingValue_q-1'] / df['Liabilities_q-1'].replace(0, np.nan)
    # ROE (Return on Equity)
    if all(col in df.columns for col in ['NetIncomeLoss_q-2', 'StockholdersEquity_q-2']):
        df['ROE'] = df['NetIncomeLoss_q-2'] / df['StockholdersEquity_q-2'].replace(0, np.nan)
    # Aggregate Insider Buy Value
    pres_col = 'Pres_Buy_Value' if 'Pres_Buy_Value' in df.columns else None
    cfo_col = 'CFO_Buy_Value' if 'CFO_Buy_Value' in df.columns else None
    agg_cols = [col for col in [pres_col, cfo_col] if col is not None]
    if agg_cols:
        df['Total_Insider_Buy_Value'] = df[agg_cols].sum(axis=1)
    # Composite Momentum
    momentum_signals = [col for col in df.columns if col.startswith('momentum_')]
    if momentum_signals:
        df['Momentum_Composite'] = df[momentum_signals].mean(axis=1)
    # Composite Trend
    trend_signals = [col for col in df.columns if col.startswith('trend_')]
    if trend_signals:
        df['Trend_Composite'] = df[trend_signals].mean(axis=1)
    # Composite Volatility
    vol_signals = [col for col in df.columns if col.startswith('volatility_')]
    if vol_signals:
        df['Volatility_Composite'] = df[vol_signals].mean(axis=1)
    # Net total cash flow
    net_cashflow_cols = [
        'NetCashProvidedByUsedInFinancingActivities_q-1',
        'NetCashProvidedByUsedInInvestingActivities_q-1',
        'NetCashProvidedByUsedInOperatingActivities_q-1'
    ]
    if all(col in df.columns for col in net_cashflow_cols):
        df['Total_Net_Cash_Flow'] = df[net_cashflow_cols].sum(axis=1)
    print("   → Aggregate and ratio features added.")

    # ---- Log transforms of heavily skewed, always-positive features ----
    log_candidates = ['Value', 'Pres_Buy_Value', 'CFO_Buy_Value', 'Qty', 'Market_SPX_Volume', 'Market_VIXY_Volume']
    for col in log_candidates:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(np.clip(df[col], 0, None))
    print("   → Log transforms added.")

    # ---- Drop any remaining constants and duplicates ----
    df = df.loc[:, df.nunique(dropna=False) > 1]
    df = df.drop_duplicates()
    
    return df, outlier_bounds
