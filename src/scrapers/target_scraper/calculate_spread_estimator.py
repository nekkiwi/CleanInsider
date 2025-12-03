import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
from ..data_loader import load_ohlcv_with_fallback
from typing import Tuple

# -----------------------------
#  Spread-estimation constants
# -----------------------------
MIN_PRICE           = 0.10       # $0.10 price floor
MIN_ADDV            = 100_000     # $10k average $ volume
MAX_SPREAD_DAY      = 0.2       # 15 % single-day cap 0.15 means almost nothing, 0.2 is almost everything
VIOLATION_SHARE_MAX = 0.50       # 50 %
# MAX_SPREAD_MED no longer used – median filter removed
MAX_INTRADAY_RNG    = 3.0        # Skip days where High / Low > 3×
# --------------------------------------------------

# Global skip counters
skip_stats = defaultdict(int)

def _calculate_corwin_schultz(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculates the Corwin-Schultz bid-ask spread estimator.
    """
    if 'High' not in df.columns or 'Low' not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    log_hl = np.log(df['High'] / df['Low'])
    log_hl_sq = log_hl**2
    
    beta = log_hl_sq.rolling(window=2).sum().rolling(window=window).mean()
    gamma = (np.log(df['High'].rolling(window=2).max() / df['Low'].rolling(window=2).min())**2).rolling(window=window).mean()

    alpha_num = (np.sqrt(2 * beta) - np.sqrt(beta))**2
    alpha_den = 3 - 2 * np.sqrt(2)
    alpha = alpha_num / alpha_den
    
    alpha[alpha < 0] = 0

    spread = 2 * (np.exp(np.sqrt(alpha)) - 1) / (1 + np.exp(np.sqrt(alpha)))
    
    return spread

def _process_ticker_for_spread(work_item: tuple, ohlcv_db_path: str) -> Tuple[pd.DataFrame | None, str]:
    """Worker returns (spread_df or None, skip_reason)"""
    ticker, group = work_item
    skip_stats['total'] += 1
    min_date = group['Filing Date'].min() - pd.Timedelta(days=60)
    
    ohlcv_df = load_ohlcv_with_fallback(ticker, ohlcv_db_path, required_start_date=min_date)
    
    if ohlcv_df.empty:
        return None, "No OHLCV data found"

    # ---------------- Data-quality / liquidity filters ----------------
    # Price floor
    if ohlcv_df['Close'].median() < MIN_PRICE:
        return None, 'penny'

    # Dollar-volume liquidity filter (60-day rolling median of ADDV)
    if 'Volume' in ohlcv_df.columns:
        dollar_vol = (ohlcv_df['Close'] * ohlcv_df['Volume']).rolling(60).mean().median()
        if pd.isna(dollar_vol) or dollar_vol < MIN_ADDV:
            return None, "Dollar volume"

    # Drop days with crazy intraday ranges
    intraday_range = ohlcv_df['High'] / ohlcv_df['Low']
    if (intraday_range > MAX_INTRADAY_RNG).any():
        return None, "Intraday range"

    ohlcv_df = ohlcv_df[intraday_range <= MAX_INTRADAY_RNG]
    if len(ohlcv_df) < 30:
        return None, "Too few rows"

    # -----------------------------------------------------------------
    spreads = _calculate_corwin_schultz(ohlcv_df)

    # Share of days whose raw CS-spread exceeds daily cap
    wide_share = (spreads > MAX_SPREAD_DAY).mean()
    if wide_share >= VIOLATION_SHARE_MAX:
        skip_stats['too_often_wide'] += 1
        return None, 'too_often_wide'

    # Keep ticker: cap extreme days
    spreads = spreads.clip(upper=MAX_SPREAD_DAY)

    skip_stats['kept'] += 1
    
    spreads.name = "corwin_schultz_spread"
    
    spread_df = spreads.reset_index()
    # Robustly find and rename the date column
    date_col = spread_df.columns[0]
    spread_df.rename(columns={date_col: 'date'}, inplace=True)
    spread_df['Ticker'] = ticker
    return spread_df, "Passed"

def generate_spread_estimates(master_events_path: Path, ohlcv_db_path: str, targets_base_path: Path, num_folds: int):
    """
    Generates Corwin-Schultz spread estimates in parallel and saves them for each fold.
    """
    print("\n--- Generating Corwin-Schultz Spread Estimates (Parallel) ---")
    
    if not master_events_path.exists():
        print(f"❌ Master events file not found at {master_events_path}. Halting.")
        return

    master_df = pd.read_parquet(master_events_path)
    master_df['Filing Date'] = pd.to_datetime(master_df['Filing Date'])
    
    # Create work items for parallel processing
    work_items = list(master_df.groupby('Ticker'))
    
    # Process in parallel
    n_jobs = max(1, min(4, len(work_items))) # Use a sensible number of cores
    tasks = [delayed(_process_ticker_for_spread)(item, ohlcv_db_path) for item in work_items]
    results = Parallel(n_jobs=n_jobs)(tqdm(tasks, desc="Calculating Spreads"))

    kept_frames = []
    for df, reason in results:
        skip_stats[reason] += 1
        if df is not None:
            kept_frames.append(df)

    # Print diagnostic summary regardless of outcome
    print("\n--- Spread generation summary ---")
    for k, v in skip_stats.items():
        print(f"{k:20}: {v}")

    if not kept_frames:
        print("❌ No spreads passed the filters. Adjust thresholds or investigate data quality.")
        return

    spreads_df = pd.concat(kept_frames)

    spreads_df['date'] = pd.to_datetime(spreads_df['date'])

    # Inform about how many tickers survived all filters
    remaining = spreads_df['Ticker'].nunique()
    total = len(work_items)
    print(f"\n✅ Spreads calculated for {remaining} tickers out of {total} after filtering.")

    # Now, for each fold, filter the master events and merge the spreads
    for i in range(1, num_folds + 2): # +1 for test set
        fold_dir_name = f"fold_{i}" if i <= num_folds else "test_set"
        fold_path = targets_base_path / fold_dir_name
        
        if not fold_path.exists():
            print(f"⚠️ Path not found: {fold_path}. Skipping.")
            continue
            
        # Determine which label file to use (training, validation, or test)
        if i <= num_folds:
            train_labels_path = fold_path / "training_labels.parquet"
            val_labels_path = fold_path / "validation_labels.parquet"
            if train_labels_path.exists():
                process_and_save_spreads(train_labels_path, spreads_df, "training_spreads.parquet")
            if val_labels_path.exists():
                process_and_save_spreads(val_labels_path, spreads_df, "validation_spreads.parquet")
        else:
            test_labels_path = fold_path / "test_labels.parquet"
            if test_labels_path.exists():
                process_and_save_spreads(test_labels_path, spreads_df, "test_spreads.parquet")

    print("\n✅ Corwin-Schultz spread estimation complete.")

def process_and_save_spreads(labels_path: Path, spreads_df: pd.DataFrame, output_filename: str):
    """
    Processes a label file, merges spreads, and saves the result.
    """
    if not labels_path.exists():
        print(f"  - Labels file not found: {labels_path.name}")
        return
        
    labels_df = pd.read_parquet(labels_path)
    labels_df['Filing Date'] = pd.to_datetime(labels_df['Filing Date'])
    
    # Merge using asof to get the latest spread on or before the filing date
    merged_df = pd.merge_asof(
        labels_df.sort_values('Filing Date'),
        spreads_df.sort_values('date'),
        left_on='Filing Date',
        right_on='date',
        by='Ticker',
        direction='backward'
    )
    
    output_path = labels_path.parent / output_filename
    
    # Keep only essential columns
    result_df = merged_df[['Ticker', 'Filing Date', 'corwin_schultz_spread']]

    # Drop rows with missing spreads or spreads beyond daily cap (2 %)
    result_df = result_df.dropna(subset=['corwin_schultz_spread'])
    result_df = result_df[result_df['corwin_schultz_spread'] <= MAX_SPREAD_DAY]
    
    result_df.to_parquet(output_path, index=False)
    print(f"  - Saved spreads for {labels_path.name} to {output_filename}")


