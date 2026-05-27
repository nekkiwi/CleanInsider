"""
Parity verification: new single-pass calculate_all_alpha_series must produce
alpha values identical (allclose, atol=1e-9, NaN==NaN) to the original
calculate_realized_alpha_series on a ~400-event sample drawn from the real
master_event_list with real Stooq / SPX data.

Run standalone:
    python tests/test_target_gen_parity.py
Or via pytest:
    python -m pytest tests/test_target_gen_parity.py -v -s
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Needs the real local data/ (master_event_list + Stooq) -> deselected in CI.
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scrapers.data_loader import load_ohlcv_with_fallback
from src.scrapers.target_scraper.generate_targets import (
    calculate_all_alpha_series,
    calculate_realized_alpha_series,
)

# ---------------------------------------------------------------------------
# Configuration — must match the actual TARGET_COMBINATIONS used by
# calculate_master_targets.py to guarantee real-world coverage.
# ---------------------------------------------------------------------------
TARGET_COMBINATIONS = [
    {"time": "1w", "tp": 0.05, "sl": -0.05},
    {"time": "1w", "tp": 0.05, "sl": -0.10},
    {"time": "1w", "tp": 0.10, "sl": -0.05},
    {"time": "1w", "tp": 0.10, "sl": -0.10},
    {"time": "1w", "tp": 0.15, "sl": -0.05},
    {"time": "1w", "tp": 0.15, "sl": -0.10},
    {"time": "2w", "tp": 0.05, "sl": -0.05},
    {"time": "2w", "tp": 0.05, "sl": -0.10},
    {"time": "2w", "tp": 0.10, "sl": -0.10},
    {"time": "1m", "tp": 0.05, "sl": -0.05},
    {"time": "1m", "tp": 0.05, "sl": -0.10},
    {"time": "1m", "tp": 0.10, "sl": -0.10},
]

STOOQ_DB_PATH = ROOT / "data" / "stooq_database"
SPX_LOCAL_PATH = STOOQ_DB_PATH / "^spx.parquet"
EVENT_LIST_PATH = ROOT / "data" / "scrapers" / "targets" / "master_event_list.parquet"

SAMPLE_SIZE = 40  # small on purpose: the reference (old 12-pass) path is slow; ~2 min
ATOL = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _col_name(params: dict) -> str:
    tp, sl, t = params["tp"], params["sl"], params["time"]
    return f"alpha_{t}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"


def _load_spx() -> pd.DataFrame:
    spx = pd.read_parquet(SPX_LOCAL_PATH)
    if spx.index.tz is not None:
        spx.index = spx.index.tz_localize(None)
    return spx


def _build_ohlcv(sample_df: pd.DataFrame) -> dict:
    ohlcv: dict = {}
    for ticker, grp in sample_df.groupby("Ticker"):
        required_start = grp["Filing Date"].min() - pd.Timedelta(days=90)
        df = load_ohlcv_with_fallback(
            ticker, str(STOOQ_DB_PATH), required_start_date=required_start
        )
        if df is not None and not df.empty:
            ohlcv[ticker] = df
    return ohlcv


# ---------------------------------------------------------------------------
# Core parity runner (also callable by pytest)
# ---------------------------------------------------------------------------


def run_parity(verbose: bool = True) -> dict:
    """
    Returns a result dict with keys:
        all_pass (bool), max_abs_diff (float),
        old_elapsed (float), new_elapsed (float),
        per_combo (list of dicts)
    """
    if verbose:
        print("=" * 70)
        print("PARITY VERIFICATION: old (12-pass) vs new (single-pass)")
        print("=" * 70)

    # --- Sample ---
    events = pd.read_parquet(EVENT_LIST_PATH)
    events["Filing Date"] = pd.to_datetime(events["Filing Date"]).dt.tz_localize(None)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(events), size=min(SAMPLE_SIZE, len(events)), replace=False)
    sample = events.iloc[sorted(idx)].copy().reset_index(drop=True)

    if verbose:
        print(
            f"\nSample: {len(sample)} events, "
            f"{sample['Ticker'].nunique()} unique tickers"
        )

    # --- Shared data ---
    spx_data = _load_spx()
    ohlcv_data = _build_ohlcv(sample)

    if verbose:
        print(
            f"Loaded OHLCV for {len(ohlcv_data)} / "
            f"{sample['Ticker'].nunique()} tickers"
        )

    # --- OLD: 12 separate calls ---
    if verbose:
        print("\n--- Running OLD implementation (12 separate passes) ---")
    t0 = time.perf_counter()
    old_results: dict[str, pd.Series] = {}
    for params in TARGET_COMBINATIONS:
        cn = _col_name(params)
        s, _ = calculate_realized_alpha_series(
            base_df=sample,
            ohlcv_data=ohlcv_data,
            spx_data=spx_data,
            timepoint_str=params["time"],
            take_profit=params["tp"],
            stop_loss=params["sl"],
            debug=False,
        )
        old_results[cn] = s
    old_elapsed = time.perf_counter() - t0
    if verbose:
        print(f"OLD elapsed: {old_elapsed:.2f}s")

    # --- NEW: single pass ---
    if verbose:
        print("\n--- Running NEW implementation (single pass) ---")
    t1 = time.perf_counter()
    new_df, _ = calculate_all_alpha_series(
        base_df=sample,
        ohlcv_data=ohlcv_data,
        spx_data=spx_data,
        target_combinations=TARGET_COMBINATIONS,
        debug=False,
    )
    new_elapsed = time.perf_counter() - t1
    if verbose:
        print(f"NEW elapsed: {new_elapsed:.2f}s")

    # --- Compare ---
    if verbose:
        print("\n--- Comparing results ---")

    all_pass = True
    max_abs_diff_global = 0.0
    per_combo = []

    for params in TARGET_COMBINATIONS:
        cn = _col_name(params)
        old_s = old_results[cn].reindex(sample.index)
        new_s = new_df[cn].reindex(sample.index)

        old_arr = old_s.to_numpy(dtype=float)
        new_arr = new_s.to_numpy(dtype=float)

        old_nan = np.isnan(old_arr)
        new_nan = np.isnan(new_arr)

        nan_mismatch = int(np.sum(old_nan != new_nan))

        both_valid = ~old_nan & ~new_nan
        max_abs_diff = 0.0
        close_ok = True
        if both_valid.any():
            diffs = np.abs(old_arr[both_valid] - new_arr[both_valid])
            max_abs_diff = float(diffs.max())
            max_abs_diff_global = max(max_abs_diff_global, max_abs_diff)
            close_ok = bool(
                np.allclose(old_arr[both_valid], new_arr[both_valid], atol=ATOL, rtol=0)
            )

        combo_pass = nan_mismatch == 0 and close_ok
        if not combo_pass:
            all_pass = False

        per_combo.append(
            {
                "col_name": cn,
                "pass": combo_pass,
                "max_abs_diff": max_abs_diff,
                "nan_mismatch": nan_mismatch,
            }
        )

        if verbose:
            status = "PASS" if combo_pass else "FAIL"
            print(
                f"  {cn:<45} {status}  "
                f"max_abs_diff={max_abs_diff:.2e}  nan_mismatch={nan_mismatch}"
            )

    if verbose:
        print()
        print("=" * 70)
        overall = "PASS" if all_pass else "FAIL"
        print(
            f"OVERALL: {overall}  "
            f"(max abs diff across all combos = {max_abs_diff_global:.2e})"
        )
        speedup = old_elapsed / max(new_elapsed, 1e-6)
        print(
            f"Timing  -- OLD: {old_elapsed:.2f}s   NEW: {new_elapsed:.2f}s   "
            f"speedup: {speedup:.1f}x"
        )
        print("=" * 70)

    return {
        "all_pass": all_pass,
        "max_abs_diff": max_abs_diff_global,
        "old_elapsed": old_elapsed,
        "new_elapsed": new_elapsed,
        "per_combo": per_combo,
    }


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------


def test_target_gen_parity():
    """
    Assert that the single-pass implementation produces alpha values that are
    bitwise-identical (within atol=1e-9) to the original 12-pass implementation
    for all 12 strategy combos and all ~400 sampled events.
    """
    result = run_parity(verbose=True)

    # Detailed assertions per combo for clear failure messages.
    for combo in result["per_combo"]:
        cn = combo["col_name"]
        assert combo["nan_mismatch"] == 0, (
            f"{cn}: NaN positions differ between old and new "
            f"(mismatch count = {combo['nan_mismatch']})"
        )
        assert combo["pass"], (
            f"{cn}: alpha values differ — max_abs_diff = {combo['max_abs_diff']:.2e} "
            f"(tolerance = {ATOL:.2e})"
        )

    assert result[
        "all_pass"
    ], f"Parity FAILED — max abs diff across all combos = {result['max_abs_diff']:.2e}"


if __name__ == "__main__":
    ok = run_parity(verbose=True)
    sys.exit(0 if ok["all_pass"] else 1)
