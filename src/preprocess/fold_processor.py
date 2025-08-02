# file: src/preprocess/fold_processor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .various_preprocessing import apply_additional_preprocessing

# The prune_globally function will be removed from here and placed directly in preprocess_features.py if needed,
# but it will only apply missingness. The rest of the pruning happens here, per-fold.


def process_fold_data(
    df: pd.DataFrame, corr_threshold: float, variance_threshold: float
) -> (pd.DataFrame, dict):
    """
    Applies the full preprocessing logic to a single fold of data.
    Order: Handle Inf -> Normalize -> Impute -> Prune Variance -> Prune Correlation -> Additional Preprocessing.
    (Note: Initial missingness filter is handled globally before this function is called).

    Args:
        df (pd.DataFrame): The data for the current fold.
        corr_threshold (float): The correlation threshold for pruning.
        variance_threshold (float): The variance threshold for pruning.

    Returns:
        pd.DataFrame: The fully preprocessed DataFrame for the fold.
        dict: The outlier bounds calculated for this specific fold.
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

    # 0. Handle infinites values (still needed here as additional preprocessing steps might introduce them)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    non_numeric_cols = df.select_dtypes(
        exclude=np.number
    ).columns  # Corrected: ensure .columns is called

    # 1. Normalize
    scaler = StandardScaler()
    # Only fit and transform numeric columns that are present and not all NaN (scaler would fail)
    cols_for_scaling = [
        col for col in numeric_cols if col in df.columns and not df[col].isnull().all()
    ]

    # Handle case where all numeric columns might be dropped or become all NaN
    if not cols_for_scaling:
        print("Warning: No numeric columns left for scaling in this fold.")
        df_normalized = pd.DataFrame(index=df.index)
    else:
        df_normalized_np = scaler.fit_transform(df[cols_for_scaling])
        df_normalized = pd.DataFrame(
            df_normalized_np, index=df.index, columns=cols_for_scaling
        )
    print("1. Normalization: Applied StandardScaler.")

    # 2. Impute (this will now handle original NaNs and the new ones from 'inf')
    df_normalized.fillna(0, inplace=True)
    print("2. Imputation: Replaced NaNs with 0.")

    # Reconstruct the DataFrame with non-numeric and normalized/imputed numeric data
    # Ensure that only columns that actually exist in df[non_numeric_cols] are selected.
    # df[non_numeric_cols.intersection(df.columns)] would be safer if non_numeric_cols isn't guaranteed to be a subset
    processed_df = pd.concat([df[non_numeric_cols], df_normalized], axis=1)

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

        if feat1 in PROTECTED_FEATURES and feat2 not in PROTECTED_FEATURES:
            next_to_drop = feat2
        elif feat2 in PROTECTED_FEATURES and feat1 not in PROTECTED_FEATURES:
            next_to_drop = feat1
        else:
            mean_corr_feat1 = corr_matrix[feat1].mean()
            mean_corr_feat2 = corr_matrix[feat2].mean()
            next_to_drop = feat1 if mean_corr_feat1 > mean_corr_feat2 else feat2

        features_to_drop_corr.add(next_to_drop)
        upper_triangle.drop(
            columns=[next_to_drop], index=[next_to_drop], inplace=True, errors="ignore"
        )

    processed_df.drop(
        columns=list(features_to_drop_corr), inplace=True, errors="ignore"
    )
    print(f"4. Correlation Filter: Dropped {len(features_to_drop_corr)} features.")

    # 5. Apply additional preprocessing (e.g., outlier clipping)
    final_df, outlier_bounds = apply_additional_preprocessing(processed_df)
    print("5. Final Preprocessing: Applied additional transformations.")

    return final_df, outlier_bounds
