"""Quick test to verify model column in logging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.alpaca.google_drive import GoogleDriveClient

print("Testing Google Sheets logging with model ID (Model as LAST column)...")

client = GoogleDriveClient()

if not client.is_connected():
    print("[FAIL] Not connected to Google Sheets")
    sys.exit(1)

# Test 1: Log a test trade with model ID
print("\n[TEST 1] Logging test trade with model ID...")
test_trade = pd.DataFrame({
    'Ticker': ['FIXED_TEST'],
    'shares': [99],
    'price': [123.45],
    'dollar_size': [999.99],
    'base_size': [0.5],
    'order_status': ['FIXED_OK']
})

result = client.log_trades(test_trade, model_id="model_TEST_FIXED")
print(f"  Trade log result: {'PASS' if result else 'FAIL'}")

# Test 2: Log test performance with model ID
print("\n[TEST 2] Logging test performance with model ID...")
test_metrics = {
    'portfolio_value': 100000,
    'cash': 50000,
    'equity': 100000,
    'num_trades': 5,
    'total_invested': 50000,
    'num_positions': 10,
    'notes': 'FIXED_TEST'
}

result = client.log_performance(test_metrics, model_id="model_TEST_FIXED")
print(f"  Performance log result: {'PASS' if result else 'FAIL'}")

print("\n" + "="*60)
print("CHECK YOUR GOOGLE SHEET:")
print("- Trade Log: FIXED_TEST ticker, 99 shares, Model=model_TEST_FIXED (LAST column)")
print("- Performance: $100,000 portfolio, Model=model_TEST_FIXED (LAST column)")
print("="*60)

