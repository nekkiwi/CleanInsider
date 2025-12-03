"""Quick Alpaca API test with immediate output."""
import sys
import os
sys.stdout.reconfigure(line_buffering=True)

print("Starting Alpaca test...")
print(f"API Key set: {'ALPACA_API_KEY' in os.environ}")
print(f"Secret set: {'ALPACA_SECRET_KEY' in os.environ}")

# Add project root
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing config...")
from src import config
print(f"  Paper mode: {config.PAPER_MODE}")
print(f"  API key prefix: {config.ALPACA_API_KEY[:8] if config.ALPACA_API_KEY else 'NOT SET'}...")

print("\nImporting alpaca-py...")
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    print("  Imports OK")
except ImportError as e:
    print(f"  Import failed: {e}")
    sys.exit(1)

print("\nCreating TradingClient...")
try:
    client = TradingClient(
        api_key=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        paper=True
    )
    print("  Client created")
except Exception as e:
    print(f"  Failed: {e}")
    sys.exit(1)

print("\nGetting account...")
try:
    account = client.get_account()
    print(f"  Status: {account.status}")
    print(f"  Portfolio: ${float(account.portfolio_value):,.2f}")
    print(f"  Cash: ${float(account.cash):,.2f}")
except Exception as e:
    print(f"  Failed: {e}")

print("\nGetting positions...")
try:
    positions = client.get_all_positions()
    print(f"  Count: {len(positions)}")
    for p in positions[:3]:
        print(f"    {p.symbol}: {p.qty} @ ${float(p.current_price):.2f}")
    if len(positions) > 3:
        print(f"    ... and {len(positions)-3} more")
except Exception as e:
    print(f"  Failed: {e}")

print("\nCreating data client...")
try:
    data_client = StockHistoricalDataClient(
        api_key=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY
    )
    print("  Data client created")
except Exception as e:
    print(f"  Failed: {e}")
    sys.exit(1)

print("\nGetting quotes for AAPL, MSFT...")
try:
    request = StockLatestQuoteRequest(symbol_or_symbols=["AAPL", "MSFT"])
    quotes = data_client.get_stock_latest_quote(request)
    for sym, quote in quotes.items():
        spread = (quote.ask_price - quote.bid_price) / quote.ask_price
        print(f"  {sym}: bid=${quote.bid_price:.2f} ask=${quote.ask_price:.2f} spread={spread:.4f}")
except Exception as e:
    print(f"  Failed: {e}")

print("\n" + "="*40)
print("TEST COMPLETE")
print("="*40)

