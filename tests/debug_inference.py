"""Quick debug for the inference pipeline."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import datetime
from src import config
from src.alpaca.inference import EnsemblePredictor
from src.alpaca.live_features import LiveFeatureGenerator
from src.alpaca.trading_client import AlpacaTradingClient
from src.alpaca.position_sizer import PositionSizer
from src.alpaca.google_drive import GoogleDriveClient

print("=" * 60)
print("Debug Inference Pipeline")
print("=" * 60)

print("\n1. Initializing clients...")
gdrive_client = GoogleDriveClient()
trading_client = AlpacaTradingClient()

print(f"   GDrive connected: {gdrive_client.is_connected()}")
print(f"   Alpaca connected: {trading_client.client is not None}")

print("\n2. Loading models...")
predictor = EnsemblePredictor()
count = predictor.load_models()
print(f"   Loaded {count} models")

print("\n3. Loading common features...")
predictor.load_common_features()
print(f"   Common features: {len(predictor.common_features)}")

print("\n4. Initializing feature generator...")
feature_generator = LiveFeatureGenerator()

print("\n5. Generating features (30 days back)...")
min_date = datetime.datetime.now() - datetime.timedelta(days=30)
features_df = feature_generator.generate_live_features(min_date=min_date)
print(f"   Features shape: {features_df.shape if not features_df.empty else 'empty'}")

if features_df.empty:
    print("\n[INFO] Checking raw insider data dates...")
    from src.scrapers.feature_scraper.scrape_openinsider import scrape_openinsider
    raw = scrape_openinsider(weeks=4)
    if not raw.empty and 'Filing Date' in raw.columns:
        print(f"   Raw dates: {raw['Filing Date'].min()} to {raw['Filing Date'].max()}")

if features_df.empty:
    print("\n[INFO] No features generated - no recent events")
    sys.exit(0)

print("\n6. Running predictions...")
signals_df = predictor.get_buy_signals(features_df)
print(f"   Signals: {len(signals_df)}")

if not signals_df.empty:
    print("\n7. Top signals:")
    print(signals_df.head(5).to_string())
    
    print("\n8. Getting spreads for top signals...")
    tickers = signals_df['Ticker'].head(5).tolist()
    spreads = trading_client.get_spreads(tickers)
    for t, s in spreads.items():
        print(f"   {t}: {s:.4f} ({s*0.5*10000:.1f} bps half)")
    
    print("\n9. Position sizing...")
    position_sizer = PositionSizer()
    account = trading_client.get_account()
    portfolio_value = float(account.get('portfolio_value', 100000))
    positions = trading_client.get_positions()
    
    sized = position_sizer.size_positions(
        signals_df.head(5),
        portfolio_value,
        positions,
        spreads
    )
    print(sized[['Ticker', 'dollar_size', 'shares', 'spread_haircut']].to_string())

print("\n" + "=" * 60)
print("Debug Complete")
print("=" * 60)

