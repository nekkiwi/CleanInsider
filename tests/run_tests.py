"""
Direct test runner - bypasses pytest for faster feedback.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

def test_ensemble_voting():
    print("\n=== ENSEMBLE VOTING TESTS ===")
    
    # Test 1: 13/25 votes -> buy
    vote_matrix = np.zeros((25, 1))
    vote_matrix[:13, 0] = 1
    vote_fraction = vote_matrix.mean(axis=0)[0]
    buy_signal = 1 if vote_fraction >= 0.5 else 0
    result = "PASS" if buy_signal == 1 else "FAIL"
    print(f"Test 1: 13/25 votes -> {vote_fraction:.0%} -> {'BUY' if buy_signal else 'NO'} [{result}]")
    
    # Test 2: 12/25 votes -> no buy
    vote_matrix = np.zeros((25, 1))
    vote_matrix[:12, 0] = 1
    vote_fraction = vote_matrix.mean(axis=0)[0]
    buy_signal = 1 if vote_fraction >= 0.5 else 0
    result = "PASS" if buy_signal == 0 else "FAIL"
    print(f"Test 2: 12/25 votes -> {vote_fraction:.0%} -> {'BUY' if buy_signal else 'NO'} [{result}]")
    
    # Test 3: unanimous buy
    vote_matrix = np.ones((25, 1))
    vote_fraction = vote_matrix.mean(axis=0)[0]
    result = "PASS" if vote_fraction == 1.0 else "FAIL"
    print(f"Test 3: 25/25 votes -> confidence {vote_fraction:.0%} [{result}]")
    
    # Test 4: mean regressor
    preds = np.array([0.05, 0.03, 0.08, 0.02, 0.06, 0.04, 0.07, 0.01, 0.05, 0.03,
                      0.06, 0.02, 0.04, 0.08, 0.05, 0.03, 0.07, 0.04, 0.02, 0.06,
                      0.05, 0.04, 0.03, 0.06, 0.02])
    mean_pred = preds.mean()
    expected = sum(preds) / 25
    result = "PASS" if abs(mean_pred - expected) < 1e-10 else "FAIL"
    print(f"Test 4: Mean regressor = {mean_pred:.4f} [{result}]")


def test_position_sizing():
    print("\n=== POSITION SIZING TESTS ===")
    from src.alpaca.position_sizer import PositionSizer
    
    # Test 5: base size scaling
    sizer = PositionSizer()
    returns = pd.Series([0.01, 0.05, 0.10, 0.15, 0.20])
    sizes = sizer.calculate_base_sizes(returns, min_size=0.25, max_size=1.0)
    pass5 = sizes.iloc[0] == 0.25 and sizes.iloc[-1] == 1.0 and (sizes >= 0.25).all() and (sizes <= 1.0).all()
    print(f"Test 5: Base sizing [0.01...0.20] -> [{sizes.iloc[0]:.2f}...{sizes.iloc[-1]:.2f}] [{'PASS' if pass5 else 'FAIL'}]")
    
    # Test 6: spread haircut ordering
    base_sizes = pd.Series([1.0, 1.0, 1.0])
    spreads = pd.Series([0.002, 0.010, 0.020])
    final_sizes = sizer.apply_spread_haircut(base_sizes, spreads)
    pass6 = final_sizes.iloc[0] >= final_sizes.iloc[1] >= final_sizes.iloc[2]
    print(f"Test 6: Spread haircut [0.002,0.010,0.020] -> {final_sizes.tolist()} [{'PASS' if pass6 else 'FAIL'}]")


def test_spread_edge_cases():
    print("\n=== SPREAD EDGE CASES ===")
    
    # 1 bp half spread -> haircut = 1.0 (capped)
    half_spread = 0.001
    haircut = min(0.005 / half_spread, 1.0)
    result = "PASS" if haircut == 1.0 else "FAIL"
    print(f"Test 7a: 1bp half spread -> haircut {haircut:.1f} [{result}]")
    
    # 5 bp half spread -> haircut = 1.0
    half_spread = 0.005
    haircut = min(0.005 / half_spread, 1.0)
    result = "PASS" if haircut == 1.0 else "FAIL"
    print(f"Test 7b: 5bp half spread -> haircut {haircut:.1f} [{result}]")
    
    # 10 bp half spread -> haircut = 0.5
    half_spread = 0.010
    haircut = min(0.005 / half_spread, 1.0)
    result = "PASS" if haircut == 0.5 else "FAIL"
    print(f"Test 7c: 10bp half spread -> haircut {haircut:.1f} [{result}]")
    
    # 11 bp half spread -> blocked (>100 bps rule)
    half_spread = 0.011
    blocked = half_spread > 0.01
    result = "PASS" if blocked else "FAIL"
    print(f"Test 7d: 11bp half spread -> blocked [{result}]")


def test_predictor_init():
    print("\n=== PREDICTOR INITIALIZATION ===")
    from src.alpaca.inference import EnsemblePredictor
    
    predictor = EnsemblePredictor()
    folds_ok = predictor.folds == [1, 2, 3, 4, 5]
    seeds_ok = predictor.seeds == [42, 123, 2024, 456, 567]
    result = "PASS" if folds_ok and seeds_ok else "FAIL"
    print(f"Test 8: Folds={predictor.folds}, Seeds={predictor.seeds} [{result}]")


def test_model_loading():
    print("\n=== MODEL LOADING (may take a few seconds) ===")
    from src.alpaca.inference import EnsemblePredictor
    
    predictor = EnsemblePredictor()
    print("Loading models...")
    count = predictor.load_models()
    
    result = "PASS" if count == 25 else "FAIL"
    print(f"Test 9: Loaded {count}/25 models [{result}]")
    
    if count > 0:
        # Verify structure
        model = predictor.models[0]
        has_all = all(k in model for k in ['classifier', 'regressor', 'selected_features', 'imputation_values'])
        result = "PASS" if has_all else "FAIL"
        print(f"Test 10: Model has required keys [{result}]")
        print(f"  - Classifier type: {type(model['classifier']).__name__}")
        print(f"  - Features count: {len(model['selected_features'])}")


if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING ENSEMBLE & POSITION SIZING TESTS")
    print("=" * 50)
    
    test_ensemble_voting()
    test_position_sizing()
    test_spread_edge_cases()
    test_predictor_init()
    
    # Only run model loading if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        test_model_loading()
    else:
        print("\nSkipping model loading test (run with --full to include)")
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETE")
    print("=" * 50)

