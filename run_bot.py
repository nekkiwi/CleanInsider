#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.live_bot_pipeline import BotArgs, run_live_bot_once
from src.alpaca.trading import execute_trades
from src.alpaca.logging import log_to_google_sheets


def run_once(timepoint: str, tp: float, sl: float, threshold_pct: int, lookback_days: int, min_signal_gate_threshold: float):
    args = BotArgs(timepoint=timepoint, tp=tp, sl=sl, threshold_pct=threshold_pct, lookback_days=lookback_days, gate=min_signal_gate_threshold)
    df = run_live_bot_once(args)
    if df is None:
        df = pd.DataFrame()
    # Execute trades and log if non-empty
    if not df.empty:
        try:
            execute_trades(df)
        except Exception:
            pass
        try:
            log_to_google_sheets(df)
        except Exception:
            pass
    return df


def main():
    parser = argparse.ArgumentParser(description='Run Insider Bot inference for a given strategy and threshold.')
    parser.add_argument('--timepoint', type=str, required=True, help='Time horizon label, e.g., 1w')
    parser.add_argument('--tp', type=float, required=True, help='Take profit percent, e.g., 0.10')
    parser.add_argument('--sl', type=float, required=True, help='Stop loss percent, e.g., -0.10')
    parser.add_argument('--threshold_pct', type=int, required=True, help='Label threshold percent (0,1,2,...)')
    args = parser.parse_args()

    lookback_days = 2
    min_signal_gate_threshold = 0.5

    df = run_once(args.timepoint, args.tp, args.sl, args.threshold_pct, lookback_days, min_signal_gate_threshold)
    outdir = Path('results') / 'bot'
    outdir.mkdir(parents=True, exist_ok=True)
    if df is not None and not df.empty:
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        fname = f"signals_{args.timepoint}_tp{str(args.tp).replace('.', 'p')}_sl{str(args.sl).replace('.', 'p')}_thr{args.threshold_pct}_{ts}.parquet"
        df.to_parquet(outdir / fname, index=False)
        print(f"Saved proposed trades to {outdir / fname}")
    else:
        print("No trades to save.")


if __name__ == '__main__':
    main()


