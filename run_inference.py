# run_inference.py
"""
Main entry point for daily inference and trading.

This script:
1. Downloads latest models from Google Drive (if configured)
2. Scrapes recent insider trading data
3. Generates features for new events
4. Runs ensemble model predictions
5. Sizes positions based on predictions and risk limits
6. Executes trades via Alpaca API (if enabled)
7. Logs results to Google Drive

Usage:
    python run_inference.py                    # Full pipeline
    python run_inference.py --dry-run          # No actual trades
    python run_inference.py --no-trade         # Generate signals only
    python run_inference.py --download-models  # Just download models
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

from src import config
from src.alpaca.inference import EnsemblePredictor
from src.alpaca.live_features import LiveFeatureGenerator
from src.alpaca.trading_client import AlpacaTradingClient
from src.alpaca.position_sizer import PositionSizer
from src.alpaca.google_drive import GoogleDriveClient


def download_models_from_drive(gdrive_client: GoogleDriveClient) -> bool:
    """Download models from Google Drive."""
    if not gdrive_client.is_connected():
        print("[WARN] Google Drive not connected, using local models")
        return False
    
    print("\n=== Downloading Models from Google Drive ===")
    strategy_str = f"{config.DEFAULT_STRATEGY[0]}_tp{str(config.DEFAULT_STRATEGY[1]).replace('.', 'p')}_sl{str(config.DEFAULT_STRATEGY[2]).replace('.', 'p')}"
    
    count = gdrive_client.download_models(strategy=strategy_str)
    return count > 0


def generate_signals(
    predictor: EnsemblePredictor,
    feature_generator: LiveFeatureGenerator,
    min_date: datetime.datetime = None
) -> pd.DataFrame:
    """Generate trading signals from live data."""
    print("\n=== Generating Trading Signals ===")
    
    # Generate live features
    print("Step 1: Scraping and generating features...")
    features_df = feature_generator.generate_live_features(min_date=min_date)
    
    if features_df.empty:
        print("[INFO] No new events found")
        return pd.DataFrame()
    
    print(f"  Found {len(features_df)} potential events")
    
    # Filter to tradable stocks
    features_df = feature_generator.filter_to_tradable(features_df)
    
    if features_df.empty:
        print("[INFO] No tradable events after filtering")
        return pd.DataFrame()
    
    # Load models if not already loaded
    if not predictor.is_loaded:
        print("Step 2: Loading ensemble models...")
        predictor.load_models()
    
    # Preprocess features
    print("Step 3: Preprocessing features...")
    scaler, _, imputation_values, _ = predictor.load_preprocessing_artifacts()
    preprocessed_df = predictor.preprocess_features(features_df, scaler, imputation_values)
    
    # Merge back identifiers
    preprocessed_df["Ticker"] = features_df["Ticker"].values
    preprocessed_df["Filing Date"] = features_df["Filing Date"].values
    if "Price" in features_df.columns:
        preprocessed_df["Price"] = features_df["Price"].values
    
    # Generate predictions
    print("Step 4: Running ensemble predictions...")
    signals_df = predictor.get_buy_signals(
        preprocessed_df,
        min_confidence=config.ENSEMBLE_VOTE_THRESHOLD
    )
    
    print(f"  Generated {len(signals_df)} buy signals")
    
    return signals_df


def size_and_execute_trades(
    signals_df: pd.DataFrame,
    trading_client: AlpacaTradingClient,
    position_sizer: PositionSizer,
    dry_run: bool = False
) -> pd.DataFrame:
    """Size positions and execute trades."""
    print("\n=== Position Sizing and Trade Execution ===")
    
    if signals_df.empty:
        print("[INFO] No signals to trade")
        return pd.DataFrame()
    
    # Get account info
    account = trading_client.get_account()
    if not account:
        print("[ERROR] Could not get account info")
        return pd.DataFrame()
    
    portfolio_value = account["portfolio_value"]
    print(f"  Portfolio value: ${portfolio_value:,.2f}")
    print(f"  Buying power: ${account['buying_power']:,.2f}")
    
    # Get current positions
    current_positions = trading_client.get_positions()
    current_exposure = sum(pos["market_value"] for pos in current_positions.values())
    print(f"  Current positions: {len(current_positions)}")
    print(f"  Current exposure: ${current_exposure:,.2f}")
    
    # Get live spreads from Alpaca for cost-adjusted sizing
    print("\nStep 1: Fetching live bid-ask spreads...")
    tickers = signals_df["Ticker"].tolist()
    spreads_dict = trading_client.get_spreads(tickers)
    spreads = pd.Series(spreads_dict)
    
    avg_spread_bps = spreads.mean() * 10000
    print(f"  Average spread: {avg_spread_bps:.1f} bps")
    
    # Also get latest prices if not in signals
    if "Price" not in signals_df.columns or signals_df["Price"].isna().any():
        prices_dict = trading_client.get_latest_prices(tickers)
        if prices_dict:
            signals_df["Price"] = signals_df["Ticker"].map(prices_dict)
    
    # Size positions with spread haircut
    print("\nStep 2: Calculating position sizes (with spread haircut)...")
    sized_df = position_sizer.size_positions(
        signals_df,
        portfolio_value,
        {k: v["market_value"] for k, v in current_positions.items()},
        spreads=spreads
    )
    
    if sized_df.empty:
        print("[INFO] No positions after sizing")
        return pd.DataFrame()
    
    print(f"  Sized {len(sized_df)} positions")
    print(f"  Total new investment: ${sized_df['dollar_size'].sum():,.2f}")
    
    # Print signal summary
    print("\n  Signals to trade:")
    for _, row in sized_df.head(10).iterrows():
        print(f"    {row['Ticker']}: {row['shares']} shares @ ${row.get('price', 0):.2f} = ${row['dollar_size']:.2f}")
    
    if len(sized_df) > 10:
        print(f"    ... and {len(sized_df) - 10} more")
    
    # Execute trades
    if dry_run:
        print("\n[DRY RUN] Skipping trade execution")
        sized_df["order_status"] = "dry_run"
        return sized_df
    
    if not trading_client.is_connected():
        print("\n[WARN] Trading client not connected, skipping execution")
        sized_df["order_status"] = "not_connected"
        return sized_df
    
    print("\nStep 3: Executing trades...")
    orders = trading_client.execute_signals(sized_df)
    
    # Map order results back to dataframe
    order_map = {o["symbol"]: o for o in orders}
    sized_df["order_id"] = sized_df["Ticker"].map(lambda t: order_map.get(t, {}).get("id"))
    sized_df["order_status"] = sized_df["Ticker"].map(lambda t: order_map.get(t, {}).get("status", "not_submitted"))
    
    print(f"\n  Submitted {len(orders)} orders")
    
    return sized_df


def log_results(
    gdrive_client: GoogleDriveClient,
    trades_df: pd.DataFrame,
    account_info: dict,
    current_positions: dict = None,
    model_id: str = "default"
):
    """Log trade results to Google Sheets."""
    print(f"\n=== Logging Results (Model: {model_id}) ===")
    
    if not gdrive_client.is_connected():
        # Save locally instead
        local_log_path = config.TRADE_LOG_PATH / f"trades_{model_id}_{datetime.date.today().isoformat()}.parquet"
        local_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not trades_df.empty:
            trades_df.to_parquet(local_log_path, index=False)
            print(f"  Saved trades locally to {local_log_path}")
        
        return
    
    # Log trades to Google Sheets with model ID
    if not trades_df.empty:
        gdrive_client.log_trades(trades_df, model_id=model_id)
    
    # Log performance to Google Sheets with model ID
    if account_info:
        metrics = {
            "portfolio_value": account_info.get("portfolio_value", 0),
            "equity": account_info.get("equity", 0),
            "cash": account_info.get("cash", 0),
            "num_trades": len(trades_df),
            "total_invested": trades_df["dollar_size"].sum() if not trades_df.empty else 0,
            "num_positions": len(current_positions) if current_positions else 0,
        }
        gdrive_client.log_performance(metrics, model_id=model_id)


def main():
    """Main entry point for inference pipeline."""
    parser = argparse.ArgumentParser(description="Run daily inference and trading")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate signals and size positions but don't execute trades"
    )
    parser.add_argument(
        "--no-trade",
        action="store_true",
        help="Only generate signals, don't size or trade"
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Only download models from Google Drive"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days to look back for insider events"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for signals (JSON or Parquet)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="1w_tp0p05_sl-0p05",
        help="Model/strategy ID for logging (e.g., model_1w_tp5_sl5)"
    )
    
    args = parser.parse_args()
    
    # Derive model ID from strategy if not explicitly set
    model_id = args.model
    
    print("=" * 60)
    print("CleanInsider Daily Inference Pipeline")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Paper' if config.PAPER_MODE else 'Live'} Trading")
    print(f"Model: {model_id}")
    print("=" * 60)
    
    # Initialize clients
    gdrive_client = GoogleDriveClient()
    trading_client = AlpacaTradingClient()
    
    # Download models only mode
    if args.download_models:
        success = download_models_from_drive(gdrive_client)
        sys.exit(0 if success else 1)
    
    # Download models if Google Drive is connected
    if gdrive_client.is_connected():
        download_models_from_drive(gdrive_client)
    
    # Initialize predictor and feature generator
    predictor = EnsemblePredictor()
    feature_generator = LiveFeatureGenerator()
    position_sizer = PositionSizer()
    
    # Load models
    try:
        num_models = predictor.load_models()
        if num_models == 0:
            print("[ERROR] No models loaded. Exiting.")
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Generate signals
    min_date = datetime.datetime.now() - datetime.timedelta(days=args.days_back)
    signals_df = generate_signals(predictor, feature_generator, min_date)
    
    # Save signals if requested
    if args.output and not signals_df.empty:
        output_path = Path(args.output)
        if output_path.suffix == ".json":
            signals_df.to_json(output_path, orient="records", indent=2)
        else:
            signals_df.to_parquet(output_path, index=False)
        print(f"\n  Saved signals to {output_path}")
    
    # Exit if no-trade mode
    if args.no_trade:
        print("\n[NO-TRADE MODE] Skipping position sizing and execution")
        if not signals_df.empty:
            print("\nTop signals:")
            print(signals_df.head(10).to_string())
        sys.exit(0)
    
    # Size and execute trades
    trades_df = size_and_execute_trades(
        signals_df,
        trading_client,
        position_sizer,
        dry_run=args.dry_run
    )
    
    # Log results to Google Sheets
    account = trading_client.get_account() or {}
    positions = trading_client.get_positions()
    log_results(gdrive_client, trades_df, account, positions, model_id=model_id)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    
    # Summary
    if not trades_df.empty:
        print(f"\nSummary:")
        print(f"  Signals generated: {len(signals_df)}")
        print(f"  Trades sized: {len(trades_df)}")
        submitted = trades_df[trades_df["order_status"].notna() & (trades_df["order_status"] != "dry_run")]
        print(f"  Orders submitted: {len(submitted)}")


if __name__ == "__main__":
    main()

