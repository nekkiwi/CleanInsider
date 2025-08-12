"""
Deprecated module: Spread estimation is now integrated into target computation
and technical features. This module remains for backward compatibility but is a no-op.
"""

import pandas as pd
from pathlib import Path


def generate_spread_estimates(master_events_path: Path, ohlcv_db_path: str, targets_base_path: Path, num_folds: int):
    print("[INFO] generate_spread_estimates is deprecated and does nothing.")
    return


def process_and_save_spreads(labels_path: Path, spreads_df: pd.DataFrame, output_filename: str):
    print("[INFO] process_and_save_spreads is deprecated and does nothing.")
    return


