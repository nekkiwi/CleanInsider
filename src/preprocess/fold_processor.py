# file: src/preprocess/fold_processor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .various_preprocessing import apply_additional_preprocessing

class FoldProcessor:
    """
    A class to handle the stateful preprocessing of a training fold and the
    consistent transformation of its corresponding evaluation fold.
    This structure prevents data leakage and ensures column consistency.
    """
    def __init__(self, corr_threshold=0.8, variance_threshold=0.0001):
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        
        # These will store the "learned" parameters from the training data
        self.scaler = StandardScaler()
        self.columns_to_scale = None
        self.imputation_values_for_unscaled = None
        self.columns_to_drop_variance = set()
        self.columns_to_drop_corr = set()
        self.final_columns = None
        self.outlier_bounds = {}

        # Constants
        self.PROTECTED_FEATURES = {
            'Ticker', 'Filing Date', 'Price', 'CommonStockSharesOutstanding_q-1', 'Value', 'Qty',
            'Pres_Buy_Value', 'CFO_Buy_Value', 'Assets_q-1', 'Liabilities_q-1', 'StockholdersEquity_q-1',
            'NetIncomeLoss_q-2', 'StockholdersEquity_q-2', 'CashAndCashEquivalentsAtCarryingValue_q-1',
            'NetCashProvidedByUsedInFinancingActivities_q-1', 'NetCashProvidedByUsedInInvestingActivities_q-1',
            'Market_SPX_Volume', 'Market_VIXY_Volume'
        }
        self.COLS_TO_SKIP_SCALING = {
            'Price', 'CEO', 'CFO', 'Pres', 'VP', 'Dir', 'TenPercent',
            'Days_Since_Trade', 'Number_of_Purchases', 'Day_Of_Year', 'Day_Of_Quarter',
            'Day_Of_Week', 'CFO_Buy_Value', 'Pres_Buy_Value', 'Insider_Importance_Score', 'Days_Since_IPO'
        }

    def fit(self, df_fit: pd.DataFrame):
        """Learns all preprocessing parameters from a fitting dataframe (the training set)."""
        print("Fitting processor on training data...")
        df = df_fit.copy().replace([np.inf, -np.inf], np.nan)

        # Step 1: Learn feature engineering parameters
        df, self.outlier_bounds = apply_additional_preprocessing(df)
        
        # Step 2: Learn which columns to scale and fit the scaler
        numeric_cols = df.select_dtypes(include=np.number).columns
        self.columns_to_scale = [col for col in numeric_cols if col not in self.COLS_TO_SKIP_SCALING and not df[col].isnull().all()]
        if self.columns_to_scale:
            self.scaler.fit(df[self.columns_to_scale].fillna(0)) # Impute with 0 for scaling
        
        # Step 3: Learn imputation values for non-scaled numeric columns
        unscaled_numeric_cols = [col for col in numeric_cols if col in self.COLS_TO_SKIP_SCALING]
        self.imputation_values_for_unscaled = df[unscaled_numeric_cols].median()

        # Step 4: Learn which columns to drop based on variance and correlation
        # Create a temporary, fully imputed numeric dataframe to check variance and correlation
        temp_df = df.select_dtypes(include=np.number).copy()
        temp_df[self.columns_to_scale] = self.scaler.transform(temp_df[self.columns_to_scale].fillna(0))
        temp_df[unscaled_numeric_cols] = temp_df[unscaled_numeric_cols].fillna(self.imputation_values_for_unscaled)
        
        # Variance filter
        cols_to_check_variance = [col for col in temp_df.columns if col not in self.PROTECTED_FEATURES]
        if cols_to_check_variance:
            variances = temp_df[cols_to_check_variance].var()
            self.columns_to_drop_variance = set(variances[variances < self.variance_threshold].index)
        
        # Correlation filter
        corr_matrix = temp_df.drop(columns=list(self.columns_to_drop_variance)).corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper_triangle.columns:
            if col in self.columns_to_drop_corr or col in self.columns_to_drop_variance: continue
            highly_corr_with = upper_triangle.index[upper_triangle[col] > self.corr_threshold].tolist()
            for correlated_feature in highly_corr_with:
                if correlated_feature in self.columns_to_drop_corr: continue
                if col in self.PROTECTED_FEATURES and correlated_feature not in self.PROTECTED_FEATURES:
                    self.columns_to_drop_corr.add(correlated_feature)
                elif correlated_feature in self.PROTECTED_FEATURES and col not in self.PROTECTED_FEATURES:
                    self.columns_to_drop_corr.add(col); break
                else:
                    self.columns_to_drop_corr.add(col if corr_matrix[col].mean() > corr_matrix[correlated_feature].mean() else correlated_feature)
        
        # Step 5: Store the final list of columns to keep
        self.final_columns = [c for c in df.columns if c not in self.columns_to_drop_variance and c not in self.columns_to_drop_corr]
        print("Processor fitting complete.")
        return self

    def transform(self, df_transform: pd.DataFrame) -> pd.DataFrame:
        """Applies the learned transformations to a new dataframe (training or evaluation set)."""
        df = df_transform.copy().replace([np.inf, -np.inf], np.nan)

        # Apply the same feature engineering and outlier clipping learned from fit()
        df, _ = apply_additional_preprocessing(df, self.outlier_bounds)

        # Apply learned scaling
        if self.columns_to_scale:
            df[self.columns_to_scale] = self.scaler.transform(df[self.columns_to_scale].fillna(0))
        
        # Apply learned imputation
        unscaled_numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col in self.COLS_TO_SKIP_SCALING]
        df[unscaled_numeric_cols] = df[unscaled_numeric_cols].fillna(self.imputation_values_for_unscaled)
        
        # Return dataframe with only the final selected columns
        # This ensures column consistency and prevents KeyErrors
        return df[[c for c in self.final_columns if c in df.columns]]

