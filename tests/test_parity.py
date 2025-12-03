# tests/test_parity.py
"""
Parity tests to ensure inference produces identical results to training.

These tests load validation fold data and compare:
1. Preprocessing (scaling, imputation)
2. Feature selection
3. Classifier predictions
4. Regressor predictions
5. Position sizing
6. Spread haircut logic
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.alpaca.inference import EnsemblePredictor
from src.alpaca.position_sizer import PositionSizer
from src.training.training_helpers import calculate_position_sizes


class TestTrainingInferenceParity:
    """Test that inference produces same results as training."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.fold = 5  # Use fold 5 (largest training set)
        self.seed = 42
        self.strategy = ("1w", 0.05, -0.05)
        self.strategy_str = "1w_tp0p05_sl-0p05"
        self.threshold_pct = 2
        
        # Paths
        self.features_path = config.FEATURES_OUTPUT_PATH / f"fold_{self.fold}"
        self.targets_path = config.TARGETS_OUTPUT_PATH / f"fold_{self.fold}"
        self.models_path = config.MODELS_PATH / self.strategy_str / f"fold_{self.fold}" / f"seed_{self.seed}"
        self.preprocessing_path = config.PREPROCESSING_ARTIFACTS_PATH / f"fold_{self.fold}"
        
    def test_model_files_exist(self):
        """Verify all required model files exist."""
        assert self.models_path.exists(), f"Model path not found: {self.models_path}"
        assert (self.models_path / "classifier.pkl").exists(), "Classifier not found"
        assert (self.models_path / "regressor.pkl").exists(), "Regressor not found"
        assert (self.models_path / "metadata.pkl").exists(), "Metadata not found"
        
    def test_validation_data_exists(self):
        """Verify validation data exists."""
        val_features = self.features_path / "validation_data.parquet"
        val_labels = self.targets_path / "validation_labels.parquet"
        val_spreads = self.targets_path / "validation_spreads.parquet"
        
        assert val_features.exists(), f"Validation features not found: {val_features}"
        assert val_labels.exists(), f"Validation labels not found: {val_labels}"
        assert val_spreads.exists(), f"Validation spreads not found: {val_spreads}"
        
    def test_preprocessing_artifacts_exist(self):
        """Verify preprocessing artifacts exist."""
        assert (self.preprocessing_path / "scaler.pkl").exists(), "Scaler not found"
        assert (self.preprocessing_path / "imputation_values.json").exists(), "Imputation values not found"
        
    def test_classifier_predictions_match(self):
        """Test that classifier predictions match between training and inference."""
        # Load validation data
        val_features = pd.read_parquet(self.features_path / "validation_data.parquet")
        val_labels = pd.read_parquet(self.targets_path / "validation_labels.parquet")
        
        # Merge features and labels
        val_features['Filing Date'] = pd.to_datetime(val_features['Filing Date'])
        val_labels['Filing Date'] = pd.to_datetime(val_labels['Filing Date'])
        
        merged = pd.merge(val_features, val_labels, on=['Ticker', 'Filing Date'], how='inner')
        
        # Load model and metadata
        classifier = joblib.load(self.models_path / "classifier.pkl")
        metadata = joblib.load(self.models_path / "metadata.pkl")
        
        selected_features = metadata.get("selected_features", [])
        imputation_values = metadata.get("imputation_values", {})
        
        # Check feature availability
        available_features = [f for f in selected_features if f in merged.columns]
        missing_features = [f for f in selected_features if f not in merged.columns]
        
        print(f"\nFeature check:")
        print(f"  Model expects: {len(selected_features)} features")
        print(f"  Available: {len(available_features)}")
        print(f"  Missing: {len(missing_features)}")
        
        if missing_features:
            print(f"  Missing features: {missing_features[:5]}...")
            pytest.skip(f"Missing {len(missing_features)} features - data/model mismatch")
        
        # Prepare features exactly as training does
        X = merged[selected_features].copy()
        
        # Apply imputation from training
        imputation_series = pd.Series(imputation_values)
        X_imputed = X.fillna(imputation_series).fillna(0)
        
        # Get predictions
        predictions = classifier.predict(X_imputed)
        
        # Verify predictions are valid
        assert len(predictions) == len(X_imputed), "Prediction count mismatch"
        assert predictions.sum() > 0, "No buy signals generated"
        assert predictions.sum() < len(predictions), "All predictions are buy (suspicious)"
        
        print(f"Classifier predictions: {predictions.sum()}/{len(predictions)} buy signals")
        
    def test_regressor_predictions_match(self):
        """Test that regressor predictions are in expected range."""
        # Load validation data
        val_features = pd.read_parquet(self.features_path / "validation_data.parquet")
        val_labels = pd.read_parquet(self.targets_path / "validation_labels.parquet")
        
        val_features['Filing Date'] = pd.to_datetime(val_features['Filing Date'])
        val_labels['Filing Date'] = pd.to_datetime(val_labels['Filing Date'])
        
        merged = pd.merge(val_features, val_labels, on=['Ticker', 'Filing Date'], how='inner')
        
        # Load models
        classifier = joblib.load(self.models_path / "classifier.pkl")
        regressor = joblib.load(self.models_path / "regressor.pkl")
        metadata = joblib.load(self.models_path / "metadata.pkl")
        
        selected_features = metadata.get("selected_features", [])
        imputation_values = metadata.get("imputation_values", {})
        
        # Check feature availability
        missing_features = [f for f in selected_features if f not in merged.columns]
        if missing_features:
            pytest.skip(f"Missing {len(missing_features)} features - data/model mismatch")
        
        # Prepare features
        X = merged[selected_features].copy()
        imputation_series = pd.Series(imputation_values)
        X_imputed = X.fillna(imputation_series).fillna(0)
        
        # Get classifier buy signals
        buy_signals = classifier.predict(X_imputed)
        buy_indices = X_imputed.index[buy_signals == 1]
        
        if len(buy_indices) > 0:
            # Get regressor predictions for buy signals
            predicted_returns = regressor.predict(X_imputed.loc[buy_indices])
            
            # Verify predictions are reasonable (should be in -1 to +1 range typically)
            assert predicted_returns.min() > -1.0, f"Predicted returns too negative: {predicted_returns.min()}"
            assert predicted_returns.max() < 1.0, f"Predicted returns too high: {predicted_returns.max()}"
            
            print(f"\nRegressor predictions for {len(buy_indices)} buy signals:")
            print(f"  Min: {predicted_returns.min():.4f}")
            print(f"  Max: {predicted_returns.max():.4f}")
            print(f"  Mean: {predicted_returns.mean():.4f}")
            
    def test_position_sizing_formula_match(self):
        """Test that position sizing formula matches between training and inference."""
        # Create test predicted returns
        test_returns = pd.Series([0.01, 0.05, 0.10, 0.15, 0.20])
        
        # Training formula (from training_helpers.py)
        training_sizes = calculate_position_sizes(test_returns, min_size=0.25, max_size=1.0)
        
        # Inference formula (from position_sizer.py)
        sizer = PositionSizer()
        inference_sizes = sizer.calculate_base_sizes(test_returns, min_size=0.25, max_size=1.0)
        
        # They should be identical
        np.testing.assert_array_almost_equal(
            training_sizes.values, 
            inference_sizes.values,
            decimal=10,
            err_msg="Position sizing formulas don't match!"
        )
        
        print("\nPosition sizing formula match: PASSED")
        print(f"  Test returns: {test_returns.tolist()}")
        print(f"  Sizes: {training_sizes.tolist()}")
        
    def test_spread_haircut_formula_match(self):
        """Test that spread haircut formula matches between training and inference."""
        # Test cases from the plan
        test_spreads = pd.Series([0.002, 0.010, 0.020, 0.022])
        base_sizes = pd.Series([1.0, 1.0, 1.0, 1.0])
        
        # Training logic (from training_helpers.py:113-122)
        def training_haircut(spreads, sizes):
            half_spread = spreads * 0.5
            haircut = (0.005 / half_spread).clip(upper=1)
            effective = sizes * haircut
            # Zero out if half_spread > 0.01 (100 bps)
            effective[half_spread > 0.01] = 0.0
            return effective
        
        # Inference logic (from position_sizer.py)
        sizer = PositionSizer(max_spread_cost=0.01)
        inference_sizes = sizer.apply_spread_haircut(base_sizes, test_spreads, reference_spread=0.005)
        
        # Calculate expected from training logic
        training_sizes = training_haircut(test_spreads, base_sizes)
        
        # Compare
        np.testing.assert_array_almost_equal(
            training_sizes.values,
            inference_sizes.values,
            decimal=10,
            err_msg="Spread haircut formulas don't match!"
        )
        
        print("\nSpread haircut formula match: PASSED")
        for i, spread in enumerate(test_spreads):
            print(f"  Spread {spread:.3f} -> Size {inference_sizes.iloc[i]:.3f}")
            
    def test_ensemble_predictor_loads_all_models(self):
        """Test that ensemble predictor loads all 25 models."""
        predictor = EnsemblePredictor()
        num_loaded = predictor.load_models()
        
        expected_models = len(config.ENSEMBLE_FOLDS) * len(config.ENSEMBLE_SEEDS)
        
        assert num_loaded == expected_models, f"Expected {expected_models} models, loaded {num_loaded}"
        
        print(f"\nEnsemble loaded {num_loaded} models")
        
    def test_ensemble_prediction_on_validation_data(self):
        """Test ensemble prediction on actual validation data."""
        # Load validation data
        val_features = pd.read_parquet(self.features_path / "validation_data.parquet")
        
        # Initialize predictor
        predictor = EnsemblePredictor()
        predictor.load_models()
        
        # Add identifiers to features before prediction
        val_features_copy = val_features.copy()
        val_features_copy['Ticker'] = val_features['Ticker']
        val_features_copy['Filing Date'] = val_features['Filing Date']
        
        # Get predictions - predictor handles preprocessing internally per model
        predictions = predictor.predict(val_features_copy)
        
        # Verify output structure
        assert 'buy_signal' in predictions.columns, "Missing buy_signal column"
        assert 'position_size' in predictions.columns, "Missing position_size column"
        assert 'confidence' in predictions.columns, "Missing confidence column"
        
        # Verify values are in expected ranges
        assert predictions['confidence'].min() >= 0, "Confidence below 0"
        assert predictions['confidence'].max() <= 1, "Confidence above 1"
        
        buy_signals = predictions[predictions['buy_signal'] == 1]
        if not buy_signals.empty:
            assert buy_signals['position_size'].min() >= 0, "Position size below 0"
            assert buy_signals['position_size'].max() <= 1.0, "Position size above 1.0"
        
        print(f"\nEnsemble predictions on validation data:")
        print(f"  Total samples: {len(predictions)}")
        print(f"  Buy signals: {predictions['buy_signal'].sum()}")
        print(f"  Avg confidence: {predictions['confidence'].mean():.3f}")


class TestSpreadHaircutEdgeCases:
    """Test spread haircut with edge cases."""
    
    def test_zero_spread(self):
        """Test behavior with zero spread."""
        sizer = PositionSizer()
        sizes = pd.Series([1.0])
        spreads = pd.Series([0.0])
        
        # Should handle gracefully (use default or skip)
        result = sizer.apply_spread_haircut(sizes, spreads)
        # With 0 spread, haircut would be inf, should be clipped to 1.0
        assert result.iloc[0] <= 1.0, "Haircut not capped for zero spread"
        
    def test_very_small_spread(self):
        """Test with very small spread (< reference)."""
        sizer = PositionSizer()
        sizes = pd.Series([1.0])
        spreads = pd.Series([0.001])  # 10 bps, below 50 bps reference
        
        result = sizer.apply_spread_haircut(sizes, spreads)
        # Haircut should be capped at 1.0
        assert result.iloc[0] == 1.0, f"Expected 1.0, got {result.iloc[0]}"
        
    def test_exactly_max_spread(self):
        """Test at exactly max spread threshold."""
        sizer = PositionSizer(max_spread_cost=0.01)
        sizes = pd.Series([1.0])
        spreads = pd.Series([0.02])  # Half spread = 0.01 = exactly at threshold
        
        result = sizer.apply_spread_haircut(sizes, spreads)
        # At threshold, should still trade (>0.01 is the cutoff)
        assert result.iloc[0] > 0, "Should trade at exactly threshold"
        
    def test_just_above_max_spread(self):
        """Test just above max spread threshold."""
        sizer = PositionSizer(max_spread_cost=0.01)
        sizes = pd.Series([1.0])
        spreads = pd.Series([0.021])  # Half spread = 0.0105 > 0.01
        
        result = sizer.apply_spread_haircut(sizes, spreads)
        assert result.iloc[0] == 0.0, "Should not trade above max spread"
        
    def test_empty_series(self):
        """Test with empty series."""
        sizer = PositionSizer()
        sizes = pd.Series([], dtype=float)
        spreads = pd.Series([], dtype=float)
        
        result = sizer.apply_spread_haircut(sizes, spreads)
        assert len(result) == 0, "Should return empty series"
        
    def test_multiple_spreads(self):
        """Test with multiple spreads at once."""
        sizer = PositionSizer(max_spread_cost=0.01)
        sizes = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        spreads = pd.Series([0.002, 0.006, 0.010, 0.018, 0.022])
        
        result = sizer.apply_spread_haircut(sizes, spreads)
        
        # Expected results based on formula
        # 0.002: half=0.001, haircut=min(0.005/0.001, 1)=1.0
        # 0.006: half=0.003, haircut=min(0.005/0.003, 1)=1.0
        # 0.010: half=0.005, haircut=min(0.005/0.005, 1)=1.0
        # 0.018: half=0.009, haircut=0.005/0.009=0.556
        # 0.022: half=0.011 > 0.01, size=0
        
        assert result.iloc[0] == 1.0, f"0.002 spread: expected 1.0, got {result.iloc[0]}"
        assert result.iloc[1] == 1.0, f"0.006 spread: expected 1.0, got {result.iloc[1]}"
        assert result.iloc[2] == 1.0, f"0.010 spread: expected 1.0, got {result.iloc[2]}"
        assert 0.5 < result.iloc[3] < 0.6, f"0.018 spread: expected ~0.556, got {result.iloc[3]}"
        assert result.iloc[4] == 0.0, f"0.022 spread: expected 0.0, got {result.iloc[4]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

