# file: src/preprocess/fold_processor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .various_preprocessing import apply_additional_preprocessing


def process_fold_data(
    df: pd.DataFrame, corr_threshold: float, variance_threshold: float
) -> (pd.DataFrame, dict):
    """
    Applies the full preprocessing logic, selectively scales variables, and
    protects pairs of correlated 'protected' features from being dropped.
    """
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

    COLS_TO_SKIP_SCALING = {
        "Price",
        "CEO",
        "CFO",
        "Pres",
        "VP",
        "Dir",
        "TenPercent",
        "Days_Since_Trade",
        "Number_of_Purchases",
        "Day_Of_Year",
        "Day_Of_Quarter",
        "Day_Of_Week",
        "CFO_Buy_Value",
        "Pres_Buy_Value",
        "Insider_Importance_Score",
        "Days_Since_IPO",
    }

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns

    # 1. Normalize (Selectively)
    scaler = StandardScaler()
    cols_to_scale = [
        col
        for col in numeric_cols
        if col not in COLS_TO_SKIP_SCALING and not df[col].isnull().all()
    ]
    unscaled_numeric_cols = [col for col in numeric_cols if col in COLS_TO_SKIP_SCALING]

    df_normalized = pd.DataFrame(index=df.index)
    if cols_to_scale:
        df_normalized_np = scaler.fit_transform(df[cols_to_scale])
        df_normalized = pd.DataFrame(
            df_normalized_np, index=df.index, columns=cols_to_scale
        )
    print("1. Normalization: Applied StandardScaler to continuous features only.")

    # 2. Impute (only on the scaled data)
    df_normalized.fillna(0, inplace=True)
    print("2. Imputation: Replaced NaNs with 0 in scaled data.")

    # Reconstruct the DataFrame
    processed_df = pd.concat(
        [df[non_numeric_cols], df[unscaled_numeric_cols], df_normalized], axis=1
    )

    # 3. Drop low-variance columns
    current_numeric_cols = processed_df.select_dtypes(include=np.number).columns
    cols_to_check_variance = [
        col for col in current_numeric_cols if col not in PROTECTED_FEATURES
    ]

    features_to_drop_variance = set()
    if len(cols_to_check_variance) > 0:
        variances = processed_df[cols_to_check_variance].var()
        features_to_drop_variance = set(variances[variances < variance_threshold].index)
        processed_df.drop(
            columns=list(features_to_drop_variance), inplace=True, errors="ignore"
        )
    print(f"3. Variance Filter: Dropped {len(features_to_drop_variance)} features.")

    # 4. Drop highly correlated columns
    numeric_df_for_corr = processed_df.select_dtypes(include=np.number)
    corr_matrix = numeric_df_for_corr.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    features_to_drop_corr = set()
    while True:
        highly_corr_pairs = (upper_triangle > corr_threshold).stack()
        if not highly_corr_pairs.any():
            break

        feat1, feat2 = highly_corr_pairs.idxmax()

        # --- NEW LOGIC: Check if both features are protected ---
        if feat1 in PROTECTED_FEATURES and feat2 in PROTECTED_FEATURES:
            # If both are protected, we must not drop either.
            # To prevent an infinite loop, set this pair's correlation to 0 in our temporary
            # upper_triangle matrix and continue to find the next most correlated pair.
            upper_triangle.loc[feat1, feat2] = 0
            print(f"   -> Preserving protected pair: ({feat1}, {feat2})")
            continue

        # --- ORIGINAL LOGIC (for all other cases) ---
        if feat1 in PROTECTED_FEATURES and feat2 not in PROTECTED_FEATURES:
            next_to_drop = feat2
        elif feat2 in PROTECTED_FEATURES and feat1 not in PROTECTED_FEATURES:
            next_to_drop = feat1
        else:  # Case where neither is protected
            mean_corr_feat1 = corr_matrix[feat1].mean()
            mean_corr_feat2 = corr_matrix[feat2].mean()
            next_to_drop = feat1 if mean_corr_feat1 > mean_corr_feat2 else feat2

        features_to_drop_corr.add(next_to_drop)
        # Drop the entire row/column for the feature we're removing from consideration
        upper_triangle.drop(
            columns=[next_to_drop], index=[next_to_drop], inplace=True, errors="ignore"
        )

    processed_df.drop(
        columns=list(features_to_drop_corr), inplace=True, errors="ignore"
    )
    print(f"4. Correlation Filter: Dropped {len(features_to_drop_corr)} features.")

    # 5. Apply additional preprocessing
    final_df, outlier_bounds = apply_additional_preprocessing(processed_df)
    print("5. Final Preprocessing: Applied additional transformations.")

    return final_df, outlier_bounds
