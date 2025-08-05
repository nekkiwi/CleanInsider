# file: src/preprocess/fold_processor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .various_preprocessing import apply_additional_preprocessing

class FoldProcessor:
    """
    A class to handle the stateful preprocessing of a training fold and the
    consistent transformation of its corresponding evaluation fold.
    """
    def __init__(self, corr_threshold=0.8, variance_threshold=0.0001):
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.columns_to_scale = None
        self.imputation_values_for_unscaled = None
        self.columns_to_drop_variance = set()
        self.columns_to_drop_corr = set()
        self.final_columns = None
        self.outlier_bounds = {}

        # --- THIS IS THE FIX: Updated Constants for yfinance annual data ---
        self.PROTECTED_FEATURES = {
            'Ticker', 'Filing Date', 'Price', 'Value', 'Qty',
            # Key Y1 Financials
            'FIN_Total Assets_Y1', 'FIN_Total Liabilities Net Minority Interest_Y1',
            'FIN_Stockholders Equity_Y1', 'FIN_Net Income_Y1', 'FIN_Total Revenue_Y1',
            # Key Y2 Financials for comparison
            'FIN_Total Assets_Y2', 'FIN_Net Income_Y2',
            # Key technicals
            'trend_sma_fast', 'momentum_rsi',
            # Key engineered features
            'FE_DebtToAssets_Y1', 'FE_ROE_Y1'
        }
        self.COLS_TO_SKIP_SCALING = {
            'Price', 'CEO', 'CFO', 'Pres', 'VP', 'Dir', 'TenPercent',
            'Days_Since_Trade', 'Number_of_Purchases', 'Day_Of_Year', 'Day_Of_Quarter',
            'Day_Of_Week'
        }
        # --- END OF FIX ---

    def fit(self, df_fit: pd.DataFrame):
        """Learns all preprocessing parameters from a fitting dataframe (the training set)."""
        print("Fitting processor on training data...")
        df = df_fit.copy().replace([np.inf, -np.inf], np.nan)
        df, self.outlier_bounds = apply_additional_preprocessing(df)
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        self.columns_to_scale = [col for col in numeric_cols if col not in self.COLS_TO_SKIP_SCALING and not df[col].isnull().all()]
        if self.columns_to_scale:
            self.scaler.fit(df[self.columns_to_scale].fillna(0))
        
        unscaled_numeric_cols = [col for col in numeric_cols if col in self.COLS_TO_SKIP_SCALING]
        self.imputation_values_for_unscaled = df[unscaled_numeric_cols].median()

        temp_df = df.select_dtypes(include=np.number).copy()
        temp_df[self.columns_to_scale] = self.scaler.transform(temp_df[self.columns_to_scale].fillna(0))
        temp_df[unscaled_numeric_cols] = temp_df[unscaled_numeric_cols].fillna(self.imputation_values_for_unscaled)
        
        cols_to_check_variance = [col for col in temp_df.columns if col not in self.PROTECTED_FEATURES]
        if cols_to_check_variance:
            variances = temp_df[cols_to_check_variance].var(ddof=0)
            self.columns_to_drop_variance = set(variances[variances < self.variance_threshold].index)
        
        corr_matrix = temp_df.drop(columns=list(self.columns_to_drop_variance)).corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        self.columns_to_drop_corr = set()
        while True:
            # Find the pair with the highest correlation
            max_corr = upper_triangle.max().max()
            if max_corr < self.corr_threshold:
                break
            
            # Get the pair of features
            s = upper_triangle.stack()
            feat1, feat2 = s.idxmax()

            # Decide which to drop
            if feat1 in self.PROTECTED_FEATURES and feat2 in self.PROTECTED_FEATURES:
                # Both protected, do nothing to them, just ignore this pair for now
                upper_triangle.loc[feat1, feat2] = 0
                continue
            elif feat1 in self.PROTECTED_FEATURES:
                self.columns_to_drop_corr.add(feat2)
                upper_triangle.drop(feat2, inplace=True, axis=1)
                upper_triangle.drop(feat2, inplace=True, axis=0)
            elif feat2 in self.PROTECTED_FEATURES:
                self.columns_to_drop_corr.add(feat1)
                upper_triangle.drop(feat1, inplace=True, axis=1)
                upper_triangle.drop(feat1, inplace=True, axis=0)
            else:
                # Neither is protected, drop the one with higher avg correlation to others
                drop_feat = feat1 if corr_matrix[feat1].mean() > corr_matrix[feat2].mean() else feat2
                self.columns_to_drop_corr.add(drop_feat)
                upper_triangle.drop(drop_feat, inplace=True, axis=1)
                upper_triangle.drop(drop_feat, inplace=True, axis=0)

        self.final_columns = [c for c in df.columns if c not in self.columns_to_drop_variance and c not in self.columns_to_drop_corr]
        print("Processor fitting complete.")
        return self

    def transform(self, df_transform: pd.DataFrame) -> pd.DataFrame:
        """Applies the learned transformations to a new dataframe."""
        df = df_transform.copy().replace([np.inf, -np.inf], np.nan)
        df, _ = apply_additional_preprocessing(df, self.outlier_bounds)

        if self.columns_to_scale:
            # Ensure the columns exist in the dataframe before trying to scale them
            cols_to_scale_in_df = [col for col in self.columns_to_scale if col in df.columns]
            df[cols_to_scale_in_df] = self.scaler.transform(df[cols_to_scale_in_df].fillna(0))
        
        unscaled_numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col in self.COLS_TO_SKIP_SCALING]
        df[unscaled_numeric_cols] = df[unscaled_numeric_cols].fillna(self.imputation_values_for_unscaled)
        
        # Return dataframe with only the final selected columns that exist in the current dataframe
        return df[[c for c in self.final_columns if c in df.columns]]

