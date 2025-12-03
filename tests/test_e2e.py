"""End-to-end test for inference pipeline."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import datetime
import pandas as pd
import numpy as np

# Force output buffering off
import functools
print = functools.partial(print, flush=True)

print("E2E Test Starting...")

# Test 1: Model loading
print("\n[TEST 1] Model Loading")
from src.alpaca.inference import EnsemblePredictor
predictor = EnsemblePredictor()
count = predictor.load_models()
print(f"  Loaded {count} models - {'PASS' if count == 25 else 'FAIL'}")

# Test 2: Load preprocessing
print("\n[TEST 2] Preprocessing Artifacts")
predictor.load_common_features()
print(f"  Common features: {len(predictor.common_features)}")

# Test 3: Create synthetic features
print("\n[TEST 3] Synthetic Prediction")
# Use one of the model's selected features as base
model = predictor.models[0]
features = model['selected_features']
print(f"  Model expects {len(features)} features")

# Create synthetic data matching expected features
np.random.seed(42)
n_samples = 3
fake_data = {}
for f in predictor.common_features:
    fake_data[f] = np.random.randn(n_samples)

fake_df = pd.DataFrame(fake_data)
fake_df['Ticker'] = ['AAPL', 'MSFT', 'GOOGL']
fake_df['Filing Date'] = datetime.datetime.now().strftime('%Y-%m-%d')

print(f"  Synthetic data shape: {fake_df.shape}")

# Test 4: Run prediction
print("\n[TEST 4] Ensemble Prediction")
signals = predictor.get_buy_signals(fake_df)
print(f"  Signals generated: {len(signals)}")
if not signals.empty:
    print(f"  Columns: {list(signals.columns)}")
    print(f"  Sample:\n{signals.head().to_string()}")
else:
    print("  No buy signals (expected on random data)")

# Test 5: Position sizing
print("\n[TEST 5] Position Sizing")
from src.alpaca.position_sizer import PositionSizer
sizer = PositionSizer()

# Create fake signals
test_signals = pd.DataFrame({
    'Ticker': ['AAPL', 'MSFT'],
    'Filing Date': ['2025-12-01', '2025-12-02'],
    'Price': [150.0, 400.0],
    'predicted_return': [0.05, 0.03],
    'vote_fraction': [0.7, 0.6],
    'position_size': [0.8, 0.5]
})

spreads = {'AAPL': 0.002, 'MSFT': 0.004}
sized = sizer.size_positions(test_signals, 100000, {}, spreads)
print(f"  Sized positions:\n{sized[['Ticker', 'dollar_size', 'shares', 'spread_haircut']].to_string()}")

# Test 6: Alpaca connection
print("\n[TEST 6] Alpaca Connection")
from src.alpaca.trading_client import AlpacaTradingClient
client = AlpacaTradingClient()
account = client.get_account()
print(f"  Portfolio: ${float(account.get('portfolio_value', 0)):,.2f}")

# Test 7: Google Sheets
print("\n[TEST 7] Google Sheets")
from src.alpaca.google_drive import GoogleDriveClient
gdrive = GoogleDriveClient()
print(f"  Connected: {gdrive.is_connected()}")

print("\n" + "=" * 50)
print("E2E TEST COMPLETE")
print("=" * 50)

