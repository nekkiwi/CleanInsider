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
    def __init__(self, corr_threshold=0.8, variance_threshold=0.0001, missing_thresh=None, debug=False):
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.columns_to_scale = None
        self.imputation_values_for_unscaled = None
        self.columns_to_drop_variance = set()
        self.columns_to_drop_corr = set()
        self.final_columns = None
        self.outlier_bounds = {}
        self.missing_thresh = missing_thresh
        self.columns_to_drop_missing = set()
        self.debug = debug  # Control debug output

        self.PROTECTED_FEATURES = {
            'Ticker', 'Filing Date', 'Price', 'Value', 'Qty',
            'FIN_Total Assets_Y1', 'FIN_Total Liabilities Net Minority Interest_Y1',
            'FIN_Stockholders Equity_Y1', 'FIN_Net Income_Y1', 'FIN_Total Revenue_Y1',
            'FIN_Total Assets_Y2', 'FIN_Net Income_Y2',
            'trend_sma_fast', 'momentum_rsi',
            'FE_DebtToAssets_Y1', 'FE_ROE_Y1'
        }
        self.COLS_TO_SKIP_SCALING = {
            'Price', 'CEO', 'CFO', 'Pres', 'VP', 'Dir', 'TenPercent',
            'Days_Since_Trade', 'Number_of_Purchases', 'Day_Of_Year', 'Day_Of_Quarter',
            'Day_Of_Week'
        }

    def _debug_print(self, message):
        """Print message only if debug mode is enabled."""
        if self.debug:
            print(message)

    def fit(self, df_fit: pd.DataFrame):
        """Learns all preprocessing parameters from a fitting dataframe (the training set)."""
        print("Fitting processor on training data...")
        df = df_fit.copy().replace([np.inf, -np.inf], np.nan)
        
        if hasattr(self, 'missing_thresh') and self.missing_thresh is not None:
            missing_proportions = df.isnull().sum() / len(df)
            cols_to_drop_missing = missing_proportions[missing_proportions >= self.missing_thresh].index.tolist()
            
            if cols_to_drop_missing:
                print(f"[PREPROCESS-INFO] Dropping {len(cols_to_drop_missing)} columns with >= {self.missing_thresh:.0%} missing values")
                self._debug_print(f"[PREPROCESS-DEBUG] Columns dropped for missingness: {cols_to_drop_missing}")
                df = df.drop(columns=cols_to_drop_missing)
                self.columns_to_drop_missing = set(cols_to_drop_missing)
            else:
                self.columns_to_drop_missing = set()
            
        df, self.outlier_bounds = apply_additional_preprocessing(df)
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        self._debug_print(f"[PREPROCESS-DEBUG] Processing {len(numeric_cols)} numeric columns for scaling")
        self._debug_print(f"[PREPROCESS-DEBUG] Columns to skip: {self.COLS_TO_SKIP_SCALING}")

        self.columns_to_scale = []
        for col in numeric_cols:
            if col not in df.columns:
                self._debug_print(f"[PREPROCESS-WARN] Column '{col}' not found in DataFrame, skipping")
                continue
            
            if col in self.COLS_TO_SKIP_SCALING:
                self._debug_print(f"[PREPROCESS-DEBUG] Skipping '{col}' (in skip list)")
                continue
            
            try:
                is_all_null = bool(df[col].isnull().all())
                if is_all_null:
                    self._debug_print(f"[PREPROCESS-DEBUG] Skipping '{col}' (all null values)")
                    continue
            except Exception as e:
                self._debug_print(f"[PREPROCESS-WARN] Error checking null status for '{col}': {e}. Skipping.")
                continue
            
            self.columns_to_scale.append(col)
            self._debug_print(f"[PREPROCESS-DEBUG] Added '{col}' to scaling list")

        print(f"[PREPROCESS-INFO] Selected {len(self.columns_to_scale)} columns for scaling")
        
        # SPECIAL HANDLING FOR PROBLEMATIC LOG COLUMNS
        log_columns = ['log_Price', 'log_Value', 'log_Qty']
        for col in log_columns:
            if col in df.columns and col in numeric_cols and col not in self.COLS_TO_SKIP_SCALING:
                try:
                    col_series = df[col].copy()
                    col_series = col_series.replace([np.inf, -np.inf], np.nan)
                    
                    valid_count = col_series.notna().sum()
                    total_count = len(col_series)
                    
                    if valid_count > 0 and valid_count / total_count > 0.1:  # At least 10% valid data
                        self.columns_to_scale.append(col)
                        print(f"[PREPROCESS-SUCCESS] Added '{col}' to scaling list ({valid_count}/{total_count} valid values)")
                    else:
                        self._debug_print(f"[PREPROCESS-SKIP] Skipping '{col}' - insufficient valid data ({valid_count}/{total_count})")
                        
                except Exception as e:
                    self._debug_print(f"[PREPROCESS-ERROR] Could not process '{col}': {e}")
        
        # Rest of the method continues...
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
        iteration_count = 0
        max_iterations = 1000

        self._debug_print(f"[PREPROCESS-DEBUG] Starting correlation filtering with threshold {self.corr_threshold}")

        while iteration_count < max_iterations:
            iteration_count += 1
            
            try:
                max_corr = float(upper_triangle.max().max())
                if pd.isna(max_corr) or max_corr < self.corr_threshold:
                    self._debug_print(f"[PREPROCESS-DEBUG] Correlation filtering complete. Max correlation: {max_corr:.4f}")
                    break
            except Exception as e:
                self._debug_print(f"[PREPROCESS-WARN] Error getting max correlation: {e}. Stopping correlation filtering.")
                break
            
            try:
                s = upper_triangle.stack()
                if s.empty:
                    self._debug_print(f"[PREPROCESS-DEBUG] No more feature pairs to process.")
                    break
                feat1, feat2 = s.idxmax()
                self._debug_print(f"[PREPROCESS-DEBUG] Processing highly correlated pair: {feat1} vs {feat2} (corr: {max_corr:.4f})")
            except Exception as e:
                self._debug_print(f"[PREPROCESS-WARN] Error finding max correlation pair: {e}. Stopping correlation filtering.")
                break

            if feat1 in self.PROTECTED_FEATURES and feat2 in self.PROTECTED_FEATURES:
                self._debug_print(f"[PREPROCESS-DEBUG] Both features protected. Ignoring pair: {feat1}, {feat2}")
                upper_triangle.loc[feat1, feat2] = 0
                continue
            elif feat1 in self.PROTECTED_FEATURES:
                drop_feat = feat2
                self._debug_print(f"[PREPROCESS-DEBUG] {feat1} is protected. Dropping {feat2}")
            elif feat2 in self.PROTECTED_FEATURES:
                drop_feat = feat1
                self._debug_print(f"[PREPROCESS-DEBUG] {feat2} is protected. Dropping {feat1}")
            else:
                try:
                    feat1_mean_corr = float(corr_matrix[feat1].mean())
                    feat2_mean_corr = float(corr_matrix[feat2].mean())
                    
                    drop_feat = feat1 if feat1_mean_corr > feat2_mean_corr else feat2
                    self._debug_print(f"[PREPROCESS-DEBUG] Correlation comparison: {feat1} ({feat1_mean_corr:.4f}) vs {feat2} ({feat2_mean_corr:.4f}). Dropping {drop_feat}")
                    
                except Exception as e:
                    self._debug_print(f"[PREPROCESS-WARN] Could not compare correlations for {feat1} vs {feat2}: {e}. Defaulting to drop {feat1}")
                    drop_feat = feat1

            try:
                self.columns_to_drop_corr.add(drop_feat)
                if drop_feat in upper_triangle.columns:
                    upper_triangle.drop(drop_feat, inplace=True, axis=1)
                if drop_feat in upper_triangle.index:
                    upper_triangle.drop(drop_feat, inplace=True, axis=0)
                self._debug_print(f"[PREPROCESS-DEBUG] Dropped {drop_feat}. Remaining features: {upper_triangle.shape[0]}")
            except Exception as e:
                self._debug_print(f"[PREPROCESS-ERROR] Error dropping feature {drop_feat}: {e}")
                break

        if iteration_count >= max_iterations:
            print(f"[PREPROCESS-WARN] Correlation filtering stopped after {max_iterations} iterations to prevent infinite loop.")

        print(f"[PREPROCESS-INFO] Correlation filtering complete. Dropped {len(self.columns_to_drop_corr)} highly correlated features.")

        self.final_columns = [c for c in df.columns if c not in self.columns_to_drop_variance and c not in self.columns_to_drop_corr]
        print("Processor fitting complete.")
        return self

    def transform(self, df_transform: pd.DataFrame) -> pd.DataFrame:
        """Applies the learned transformations to a new dataframe."""
        df = df_transform.copy().replace([np.inf, -np.inf], np.nan)
        df, _ = apply_additional_preprocessing(df, self.outlier_bounds)

        if self.columns_to_scale:
            cols_to_scale_in_df = [col for col in self.columns_to_scale if col in df.columns]
            df[cols_to_scale_in_df] = self.scaler.transform(df[cols_to_scale_in_df].fillna(0))
        
        unscaled_numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col in self.COLS_TO_SKIP_SCALING]
        df[unscaled_numeric_cols] = df[unscaled_numeric_cols].fillna(self.imputation_values_for_unscaled)
        
        return df[[c for c in self.final_columns if c in df.columns]]
