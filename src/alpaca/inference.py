# src/alpaca/inference.py
"""
Ensemble model inference for generating trading signals.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config


class EnsemblePredictor:
    """
    Loads and runs ensemble predictions using multiple models across folds and seeds.
    
    Uses majority voting for classification and mean for regression predictions.
    """
    
    def __init__(
        self,
        strategy: tuple = None,
        folds: list = None,
        seeds: list = None,
        models_base_path: Path = None,
        preprocessing_path: Path = None
    ):
        """
        Initialize the ensemble predictor.
        
        Args:
            strategy: Tuple of (timepoint, take_profit, stop_loss)
            folds: List of fold numbers to include in ensemble
            seeds: List of random seeds to include in ensemble
            models_base_path: Path to saved models directory
            preprocessing_path: Path to preprocessing artifacts
        """
        self.strategy = strategy or config.DEFAULT_STRATEGY
        self.folds = folds or config.ENSEMBLE_FOLDS
        self.seeds = seeds or config.ENSEMBLE_SEEDS
        self.models_base_path = models_base_path or config.MODELS_PATH
        self.preprocessing_path = preprocessing_path or config.PREPROCESSING_ARTIFACTS_PATH
        
        self.models = []
        self.common_features = None
        self.is_loaded = False
        
    def _get_strategy_string(self) -> str:
        """Convert strategy tuple to folder-safe string."""
        timepoint, tp, sl = self.strategy
        return f"{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"
    
    def load_common_features(self) -> list:
        """Load the list of features common across all training folds."""
        common_features_path = self.preprocessing_path / "common_features.json"
        if not common_features_path.exists():
            raise FileNotFoundError(f"Common features file not found: {common_features_path}")
        
        with open(common_features_path, "r") as f:
            self.common_features = json.load(f)
        
        return self.common_features
    
    def load_models(self) -> int:
        """
        Load all models in the ensemble.
        
        Returns:
            Number of models successfully loaded
        """
        if self.common_features is None:
            self.load_common_features()
            
        strategy_str = self._get_strategy_string()
        strategy_path = self.models_base_path / strategy_str
        
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy models not found: {strategy_path}")
        
        self.models = []
        
        for fold in self.folds:
            for seed in self.seeds:
                model_dir = strategy_path / f"fold_{fold}" / f"seed_{seed}"
                
                if not model_dir.exists():
                    print(f"[WARN] Model directory not found: {model_dir}")
                    continue
                
                try:
                    classifier_path = model_dir / "classifier.pkl"
                    regressor_path = model_dir / "regressor.pkl"
                    metadata_path = model_dir / "metadata.pkl"
                    
                    classifier = joblib.load(classifier_path) if classifier_path.exists() else None
                    regressor = joblib.load(regressor_path) if regressor_path.exists() else None
                    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
                    
                    if classifier is not None:
                        self.models.append({
                            "fold": fold,
                            "seed": seed,
                            "classifier": classifier,
                            "regressor": regressor,
                            "selected_features": metadata.get("selected_features", []),
                            "imputation_values": metadata.get("imputation_values", {})
                        })
                except Exception as e:
                    print(f"[ERROR] Failed to load model {model_dir}: {e}")
                    continue
        
        self.is_loaded = len(self.models) > 0
        print(f"[INFO] Loaded {len(self.models)} models for ensemble")
        return len(self.models)
    
    def load_preprocessing_artifacts(self, fold: int = 5) -> tuple:
        """
        Load preprocessing artifacts for a specific fold.
        
        Args:
            fold: Fold number to load artifacts from (default: 5, the largest training set)
            
        Returns:
            Tuple of (scaler, outlier_bounds, imputation_values, columns_info)
        """
        fold_dir = self.preprocessing_path / f"fold_{fold}"
        
        if not fold_dir.exists():
            raise FileNotFoundError(f"Preprocessing artifacts not found: {fold_dir}")
        
        scaler_path = fold_dir / "scaler.pkl"
        outlier_path = fold_dir / "outlier_bounds.json"
        imputation_path = fold_dir / "imputation_values.json"
        columns_path = fold_dir / "preprocessing_columns.json"
        
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        outlier_bounds = {}
        if outlier_path.exists():
            with open(outlier_path, "r") as f:
                outlier_bounds = json.load(f)
        
        imputation_values = {}
        if imputation_path.exists():
            with open(imputation_path, "r") as f:
                imputation_values = json.load(f)
        
        columns_info = {}
        if columns_path.exists():
            with open(columns_path, "r") as f:
                columns_info = json.load(f)
        
        return scaler, outlier_bounds, imputation_values, columns_info
    
    def preprocess_features(
        self,
        features_df: pd.DataFrame,
        scaler: StandardScaler = None,
        imputation_values: dict = None
    ) -> pd.DataFrame:
        """
        Apply preprocessing transformations to feature dataframe.
        
        Args:
            features_df: Raw feature dataframe
            scaler: Fitted StandardScaler (optional, will load if not provided)
            imputation_values: Imputation values dict (optional)
            
        Returns:
            Preprocessed feature dataframe
        """
        if scaler is None or imputation_values is None:
            scaler, _, imputation_values, columns_info = self.load_preprocessing_artifacts()
        
        df = features_df.copy()
        
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Filter to common features only
        available_features = [col for col in self.common_features if col in df.columns]
        df = df[available_features]
        
        # Apply imputation
        for col, val in imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        
        # Apply scaling if scaler is available
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            scale_cols = [col for col in scaler.feature_names_in_ if col in df.columns]
            if scale_cols:
                df[scale_cols] = scaler.transform(df[scale_cols].fillna(0))
        
        return df
    
    def predict(
        self,
        features_df: pd.DataFrame,
        vote_threshold: float = None
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions for the given features.
        
        Args:
            features_df: Preprocessed feature dataframe (must have Ticker, Filing Date)
            vote_threshold: Fraction of models that must agree for a buy signal
            
        Returns:
            DataFrame with columns: Ticker, Filing Date, buy_signal, position_size, confidence
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if vote_threshold is None:
            vote_threshold = config.ENSEMBLE_VOTE_THRESHOLD
        
        # Store identifiers
        identifiers = features_df[["Ticker", "Filing Date"]].copy()
        
        # Collect predictions from all models
        all_buy_votes = []
        all_predicted_returns = []
        
        for model_info in self.models:
            classifier = model_info["classifier"]
            regressor = model_info["regressor"]
            selected_features = model_info["selected_features"]
            imputation_values = model_info["imputation_values"]
            
            # Get features this model was trained on
            available_features = [f for f in selected_features if f in features_df.columns]
            
            if not available_features:
                continue
            
            X = features_df[available_features].copy()
            
            # Apply model-specific imputation
            for col, val in imputation_values.items():
                if col in X.columns:
                    X[col] = X[col].fillna(val)
            
            X = X.fillna(0)
            
            # Get classifier predictions
            try:
                buy_votes = classifier.predict(X)
                all_buy_votes.append(buy_votes)
                
                # Get regressor predictions for buy signals
                if regressor is not None:
                    predicted_returns = np.zeros(len(X))
                    buy_mask = buy_votes == 1
                    if buy_mask.any():
                        predicted_returns[buy_mask] = regressor.predict(X[buy_mask])
                    all_predicted_returns.append(predicted_returns)
            except Exception as e:
                print(f"[WARN] Prediction failed for model fold={model_info['fold']}, seed={model_info['seed']}: {e}")
                continue
        
        if not all_buy_votes:
            return pd.DataFrame({
                "Ticker": identifiers["Ticker"],
                "Filing Date": identifiers["Filing Date"],
                "buy_signal": 0,
                "position_size": 0.0,
                "confidence": 0.0
            })
        
        # Ensemble voting for buy signal
        vote_matrix = np.array(all_buy_votes)
        vote_fractions = vote_matrix.mean(axis=0)
        buy_signals = (vote_fractions >= vote_threshold).astype(int)
        
        # Average predicted returns across models
        if all_predicted_returns:
            returns_matrix = np.array(all_predicted_returns)
            mean_returns = returns_matrix.mean(axis=0)
        else:
            mean_returns = np.zeros(len(features_df))
        
        # Calculate position sizes (0.25 to 1.0 range)
        position_sizes = np.zeros(len(features_df))
        buy_mask = buy_signals == 1
        
        if buy_mask.any():
            pos_returns = mean_returns[buy_mask]
            if pos_returns.max() != pos_returns.min():
                scaled = (pos_returns - pos_returns.min()) / (pos_returns.max() - pos_returns.min())
                position_sizes[buy_mask] = 0.25 + scaled * 0.75
            else:
                position_sizes[buy_mask] = 0.625  # Midpoint
        
        results = pd.DataFrame({
            "Ticker": identifiers["Ticker"],
            "Filing Date": identifiers["Filing Date"],
            "buy_signal": buy_signals,
            "position_size": position_sizes,
            "confidence": vote_fractions,
            "predicted_return": mean_returns
        })
        
        return results
    
    def get_buy_signals(
        self,
        features_df: pd.DataFrame,
        vote_threshold: float = None,
        min_confidence: float = 0.5
    ) -> pd.DataFrame:
        """
        Get only the buy signals from predictions.
        
        Args:
            features_df: Preprocessed feature dataframe
            vote_threshold: Fraction of models that must agree
            min_confidence: Minimum confidence to include
            
        Returns:
            DataFrame of buy signals only
        """
        predictions = self.predict(features_df, vote_threshold)
        
        buy_signals = predictions[
            (predictions["buy_signal"] == 1) & 
            (predictions["confidence"] >= min_confidence)
        ].copy()
        
        return buy_signals.sort_values("predicted_return", ascending=False)


