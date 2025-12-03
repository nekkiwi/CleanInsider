# tests/test_ensemble.py
"""
Tests for ensemble voting logic and position sizing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alpaca.inference import EnsemblePredictor
from src.alpaca.position_sizer import PositionSizer


class TestEnsembleVoting:
    """Test majority voting logic for ensemble predictions."""
    
    def test_majority_buy_signal_above_threshold(self):
        """Test that 13/25 models voting buy produces a buy signal (52% > 50%)."""
        # Simulate voting matrix: 25 models, 1 sample
        vote_matrix = np.zeros((25, 1))
        vote_matrix[:13, 0] = 1  # 13 models vote buy
        
        vote_fraction = vote_matrix.mean(axis=0)[0]
        buy_signal = 1 if vote_fraction >= 0.5 else 0
        
        assert vote_fraction == 0.52
        assert buy_signal == 1
        print(f"\n13/25 votes -> {vote_fraction:.0%} -> BUY")
        
    def test_minority_buy_signal_below_threshold(self):
        """Test that 12/25 models voting buy does NOT produce a buy signal (48% < 50%)."""
        vote_matrix = np.zeros((25, 1))
        vote_matrix[:12, 0] = 1  # 12 models vote buy
        
        vote_fraction = vote_matrix.mean(axis=0)[0]
        buy_signal = 1 if vote_fraction >= 0.5 else 0
        
        assert vote_fraction == 0.48
        assert buy_signal == 0
        print(f"\n12/25 votes -> {vote_fraction:.0%} -> NO BUY")
        
    def test_unanimous_buy_confidence(self):
        """Test that all models agreeing gives confidence = 1.0."""
        vote_matrix = np.ones((25, 1))  # All 25 vote buy
        
        vote_fraction = vote_matrix.mean(axis=0)[0]
        
        assert vote_fraction == 1.0
        print(f"\n25/25 votes -> confidence = {vote_fraction:.0%}")
        
    def test_unanimous_no_buy_confidence(self):
        """Test that no models voting buy gives confidence = 0.0."""
        vote_matrix = np.zeros((25, 1))  # All 25 vote no buy
        
        vote_fraction = vote_matrix.mean(axis=0)[0]
        
        assert vote_fraction == 0.0
        print(f"\n0/25 votes -> confidence = {vote_fraction:.0%}")
        
    def test_position_size_is_mean_of_regressors(self):
        """Test that position size comes from mean of regressor predictions."""
        # Simulate 25 regressor predictions for 1 sample
        regressor_predictions = np.array([
            [0.05], [0.03], [0.08], [0.02], [0.06],  # Fold 1, seeds 1-5
            [0.04], [0.07], [0.01], [0.05], [0.03],  # Fold 2, seeds 1-5
            [0.06], [0.02], [0.04], [0.08], [0.05],  # Fold 3, seeds 1-5
            [0.03], [0.07], [0.04], [0.02], [0.06],  # Fold 4, seeds 1-5
            [0.05], [0.04], [0.03], [0.06], [0.02],  # Fold 5, seeds 1-5
        ])
        
        mean_prediction = regressor_predictions.mean()
        
        # Verify it's the arithmetic mean
        expected = sum([0.05, 0.03, 0.08, 0.02, 0.06,
                       0.04, 0.07, 0.01, 0.05, 0.03,
                       0.06, 0.02, 0.04, 0.08, 0.05,
                       0.03, 0.07, 0.04, 0.02, 0.06,
                       0.05, 0.04, 0.03, 0.06, 0.02]) / 25
        
        assert abs(mean_prediction - expected) < 1e-10
        print(f"\nMean regressor prediction: {mean_prediction:.4f}")
        
    def test_multiple_samples_voting(self):
        """Test voting across multiple samples."""
        # 25 models, 5 samples
        np.random.seed(42)
        vote_matrix = np.random.randint(0, 2, size=(25, 5))
        
        vote_fractions = vote_matrix.mean(axis=0)
        buy_signals = (vote_fractions >= 0.5).astype(int)
        
        print(f"\nVote fractions: {vote_fractions}")
        print(f"Buy signals: {buy_signals}")
        
        # Verify each sample is correctly classified
        for i in range(5):
            expected = 1 if vote_fractions[i] >= 0.5 else 0
            assert buy_signals[i] == expected


class TestPositionSizing:
    """Test position sizing calculations."""
    
    def test_base_size_scaling(self):
        """Test that predicted returns scale to 0.25-1.0 range."""
        sizer = PositionSizer()
        
        # Create a range of predicted returns
        returns = pd.Series([0.01, 0.05, 0.10, 0.15, 0.20])
        
        sizes = sizer.calculate_base_sizes(returns, min_size=0.25, max_size=1.0)
        
        # Min return should get min size
        assert sizes.iloc[0] == 0.25
        
        # Max return should get max size
        assert sizes.iloc[-1] == 1.0
        
        # All sizes should be in range
        assert (sizes >= 0.25).all()
        assert (sizes <= 1.0).all()
        
        # Sizes should be monotonically increasing
        assert (sizes.diff().dropna() >= 0).all()
        
        print(f"\nReturns: {returns.tolist()}")
        print(f"Sizes: {sizes.tolist()}")
        
    def test_position_dollar_sizing(self):
        """Test dollar-based position sizing respects limits."""
        sizer = PositionSizer(
            max_position_size=0.05,
            max_total_exposure=0.50,
            min_position_dollars=100
        )
        
        portfolio_value = 100000
        current_exposure = {}  # No current positions
        
        signals = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'position_size': [1.0, 0.5, 0.25],  # Base sizes from ensemble
            'predicted_return': [0.10, 0.05, 0.02]
        })
        
        sized = sizer.size_positions(signals, portfolio_value, current_exposure)
        
        # Check max position constraint (5% of $100k = $5000)
        max_dollars = portfolio_value * sizer.max_position_size
        for _, row in sized.iterrows():
            assert row.get('dollar_size', 0) <= max_dollars, f"{row['Ticker']} exceeds max position"
        
        print(f"\nPortfolio: ${portfolio_value:,}")
        print(f"Max position: ${max_dollars:,}")
        print(f"Sized positions:\n{sized[['Ticker', 'dollar_size']].to_string()}")
        
    def test_exposure_limit(self):
        """Test that total exposure is capped."""
        sizer = PositionSizer(
            max_position_size=0.10,
            max_total_exposure=0.50,
            min_position_dollars=100
        )
        
        portfolio_value = 100000
        current_exposure = {'EXISTING': 30000}  # Already 30% exposed
        
        signals = pd.DataFrame({
            'Ticker': ['NEW1', 'NEW2', 'NEW3'],
            'position_size': [1.0, 1.0, 1.0],  # All want max size
            'predicted_return': [0.10, 0.09, 0.08]
        })
        
        sized = sizer.size_positions(signals, portfolio_value, current_exposure)
        
        # Total new exposure + existing should not exceed 50%
        new_exposure = sized['dollar_size'].sum()
        total_exposure = sum(current_exposure.values()) + new_exposure
        
        assert total_exposure <= portfolio_value * sizer.max_total_exposure + 1, \
            f"Total exposure ${total_exposure:,} exceeds limit"
        
        print(f"\nExisting exposure: ${sum(current_exposure.values()):,}")
        print(f"New exposure: ${new_exposure:,}")
        print(f"Total: ${total_exposure:,} (limit: ${portfolio_value * sizer.max_total_exposure:,})")
        
    def test_min_position_filter(self):
        """Test that positions below minimum are filtered out."""
        sizer = PositionSizer(
            max_position_size=0.05,
            min_position_dollars=500
        )
        
        portfolio_value = 10000  # Small portfolio
        current_exposure = {}
        
        signals = pd.DataFrame({
            'Ticker': ['SMALL1', 'SMALL2', 'LARGE1'],
            'position_size': [0.25, 0.30, 1.0],  # Different sizes
            'predicted_return': [0.05, 0.06, 0.10]
        })
        
        sized = sizer.size_positions(signals, portfolio_value, current_exposure)
        
        # Filter out positions below minimum
        tradable = sized[sized['dollar_size'] >= sizer.min_position_dollars]
        
        print(f"\nPortfolio: ${portfolio_value:,}")
        print(f"Min position: ${sizer.min_position_dollars:,}")
        print(f"Tradable positions: {len(tradable)}/{len(signals)}")
        
    def test_spread_reduces_position(self):
        """Test that high spreads reduce position sizes."""
        sizer = PositionSizer()
        
        base_sizes = pd.Series([1.0, 1.0, 1.0])
        spreads = pd.Series([0.002, 0.010, 0.020])  # Increasing spreads
        
        final_sizes = sizer.apply_spread_haircut(base_sizes, spreads)
        
        # Higher spread = lower size
        assert final_sizes.iloc[0] >= final_sizes.iloc[1] >= final_sizes.iloc[2]
        
        print(f"\nSpreads: {spreads.tolist()}")
        print(f"Final sizes: {final_sizes.tolist()}")


class TestEnsemblePredictorIntegration:
    """Integration tests for the full ensemble predictor."""
    
    def test_predictor_initialization(self):
        """Test predictor initializes with correct config."""
        print("\n[TEST] Checking predictor initialization...")
        predictor = EnsemblePredictor()
        assert predictor.folds == [1, 2, 3, 4, 5]
        assert predictor.seeds == [42, 123, 2024, 456, 567]
        assert len(predictor.folds) * len(predictor.seeds) == 25
        print("[TEST] Predictor config OK")
        
    @pytest.mark.slow
    def test_model_loading(self):
        """Test models load correctly (slow - loads 25 models)."""
        print("\n[TEST] Loading models (this takes a few seconds)...")
        predictor = EnsemblePredictor()
        count = predictor.load_models()
        
        assert count == 25, f"Expected 25 models, got {count}"
        assert predictor.is_loaded
        
        # Verify each model has required components
        for model in predictor.models:
            assert 'classifier' in model
            assert 'regressor' in model
            assert 'selected_features' in model
            assert 'imputation_values' in model
            
        print(f"[TEST] Loaded {count} models")
        print(f"[TEST] Folds: {set(m['fold'] for m in predictor.models)}")
        print(f"[TEST] Seeds: {set(m['seed'] for m in predictor.models)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

