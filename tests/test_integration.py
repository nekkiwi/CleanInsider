"""
Integration tests for Alpaca and Google Sheets APIs.
These tests verify API connectivity and basic operations.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import pandas as pd
from datetime import datetime


def test_alpaca_connection():
    """Test Alpaca API connectivity and account info."""
    print("\n=== ALPACA CONNECTION TEST ===")
    
    try:
        from src.alpaca.trading_client import AlpacaTradingClient
        
        client = AlpacaTradingClient()
        print(f"Paper mode: {client.paper_mode}")
        
        # Test account info
        account = client.get_account()
        print(f"Account status: {account.get('status', 'unknown')}")
        print(f"Portfolio value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"Buying power: ${float(account.get('buying_power', 0)):,.2f}")
        print("[PASS] Alpaca connection OK")
        return True
        
    except Exception as e:
        print(f"[FAIL] Alpaca connection failed: {e}")
        return False


def test_alpaca_positions():
    """Test fetching current positions."""
    print("\n=== ALPACA POSITIONS TEST ===")
    
    try:
        from src.alpaca.trading_client import AlpacaTradingClient
        
        client = AlpacaTradingClient()
        positions = client.get_positions()  # Returns Dict[symbol, info]
        
        print(f"Current positions: {len(positions)}")
        for i, (symbol, pos) in enumerate(positions.items()):
            if i >= 5:
                print(f"  ... and {len(positions) - 5} more")
                break
            print(f"  {symbol}: {pos.get('qty')} shares @ ${float(pos.get('current_price', 0)):.2f}")
        
        print("[PASS] Positions fetch OK")
        return True
        
    except Exception as e:
        print(f"[FAIL] Positions fetch failed: {e}")
        return False


def test_alpaca_spreads():
    """Test fetching live spreads from Alpaca."""
    print("\n=== ALPACA SPREADS TEST ===")
    
    try:
        from src.alpaca.trading_client import AlpacaTradingClient
        
        client = AlpacaTradingClient()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        spreads = client.get_spreads(symbols)
        
        print(f"Spreads fetched for {len(spreads)} symbols:")
        for sym, spread in spreads.items():
            half_spread_bps = spread * 0.5 * 10000
            print(f"  {sym}: {spread:.4f} (half spread: {half_spread_bps:.1f} bps)")
        
        print("[PASS] Spreads fetch OK")
        return True
        
    except Exception as e:
        print(f"[FAIL] Spreads fetch failed: {e}")
        return False


def test_google_sheets_connection():
    """Test Google Sheets API connectivity."""
    print("\n=== GOOGLE SHEETS CONNECTION TEST ===")
    
    try:
        from src.alpaca.google_drive import GoogleDriveClient
        
        client = GoogleDriveClient()
        
        if client.sheets_service is None:
            print("[SKIP] Google Sheets not configured (no credentials)")
            return None
        
        print(f"Sheet ID: {client.log_sheet_id[:20]}..." if client.log_sheet_id else "No sheet ID")
        print("[PASS] Google Sheets connection OK")
        return True
        
    except Exception as e:
        print(f"[FAIL] Google Sheets connection failed: {e}")
        return False


def test_google_sheets_logging():
    """Test logging trades to Google Sheets."""
    print("\n=== GOOGLE SHEETS LOGGING TEST ===")
    
    try:
        from src.alpaca.google_drive import GoogleDriveClient
        
        client = GoogleDriveClient()
        
        if client.sheets_service is None:
            print("[SKIP] Google Sheets not configured")
            return None
        
        # Create test trade data
        test_trades = pd.DataFrame({
            'timestamp': [datetime.now().isoformat()],
            'ticker': ['TEST_TICKER'],
            'action': ['TEST_BUY'],
            'shares': [0],
            'price': [0.0],
            'dollar_size': [0.0],
            'predicted_return': [0.0],
            'vote_fraction': [0.0],
            'spread': [0.0],
            'status': ['TEST - IGNORE']
        })
        
        result = client.log_trades(test_trades)
        
        if result:
            print("Test trade logged successfully")
            print("[PASS] Google Sheets logging OK")
        else:
            print("[FAIL] Logging returned False")
            
        return result
        
    except Exception as e:
        print(f"[FAIL] Google Sheets logging failed: {e}")
        return False


def test_google_drive_models():
    """Test downloading models from Google Drive."""
    print("\n=== GOOGLE DRIVE MODELS TEST ===")
    
    try:
        from src.alpaca.google_drive import GoogleDriveClient
        
        client = GoogleDriveClient()
        
        if client.drive_service is None:
            print("[SKIP] Google Drive not configured")
            return None
        
        # List files in models folder
        folder_id = os.environ.get("GDRIVE_MODELS_FOLDER_ID")
        if not folder_id:
            print("[SKIP] No models folder ID configured")
            return None
        
        # Try to list files
        results = client.drive_service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name, mimeType)",
            pageSize=10
        ).execute()
        
        files = results.get('files', [])
        print(f"Found {len(files)} items in models folder:")
        for f in files[:5]:
            print(f"  {f['name']} ({f['mimeType'].split('.')[-1]})")
        
        print("[PASS] Google Drive models folder accessible")
        return True
        
    except Exception as e:
        print(f"[FAIL] Google Drive models test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 50)
    
    results = {}
    
    # Alpaca tests
    results['alpaca_conn'] = test_alpaca_connection()
    results['alpaca_pos'] = test_alpaca_positions()
    results['alpaca_spreads'] = test_alpaca_spreads()
    
    # Google tests
    results['gsheets_conn'] = test_google_sheets_connection()
    results['gsheets_log'] = test_google_sheets_logging()
    results['gdrive_models'] = test_google_drive_models()
    
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    for name, result in results.items():
        status = "PASS" if result is True else ("SKIP" if result is None else "FAIL")
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

