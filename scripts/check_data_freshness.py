#!/usr/bin/env python
"""
Report (and optionally gate on) the freshness of the data artifacts.

Used two ways:
  * Manually after a data refresh to confirm everything is current.
  * By the scheduled `data_freshness` GitHub workflow to flag staleness.

Exit code is non-zero when any checked artifact is older than --max-stale-days,
so it can fail a CI job.

Usage:
    python scripts/check_data_freshness.py
    python scripts/check_data_freshness.py --max-stale-days 30
"""

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import config  # noqa: E402


def _max_date(parquet_path: Path, date_col: str = "Filing Date"):
    """Return the max date in a parquet file's date column, or None."""
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=[date_col])
    except Exception:
        df = pd.read_parquet(parquet_path)
        if date_col not in df.columns:
            return None
    s = pd.to_datetime(df[date_col], errors="coerce")
    return s.max()


def _latest_sec_quarter(quarter_dir: Path):
    """Return the latest SEC quarter zip stem (e.g. '2025q2'), or None."""
    if not quarter_dir.exists():
        return None
    zips = sorted(p.stem for p in quarter_dir.glob("*.zip"))
    return zips[-1] if zips else None


def main():
    parser = argparse.ArgumentParser(description="Check data artifact freshness.")
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=30,
        help="Fail if the newest event date is older than this many days.",
    )
    args = parser.parse_args()

    today = pd.Timestamp(datetime.date.today())
    features_dir = Path(config.FEATURES_OUTPUT_PATH)
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)

    checks = {
        "openinsider_data": features_dir / "components" / "openinsider_data.parquet",
        "raw_features": features_dir / "raw_features.parquet",
        "master_event_list": targets_dir / "master_event_list.parquet",
        "test_set": features_dir / "test_set" / "test_data.parquet",
    }

    print("=" * 64)
    print(
        f"DATA FRESHNESS CHECK  (today={today.date()}, max_stale={args.max_stale_days}d)"
    )
    print("=" * 64)

    stale = []
    missing = []
    for name, path in checks.items():
        if not path.exists():
            print(f"  [MISSING] {name:18s} -> {path}")
            missing.append(name)
            continue
        mx = _max_date(path)
        if mx is None or pd.isna(mx):
            print(f"  [NO DATE] {name:18s} (could not read a date column)")
            missing.append(name)
            continue
        age = (today - mx).days
        flag = "STALE" if age > args.max_stale_days else "ok"
        print(f"  [{flag:5s}]   {name:18s} max={mx.date()}  ({age}d old)")
        if age > args.max_stale_days:
            stale.append((name, age))

    sec_q = _latest_sec_quarter(Path(config.EDGAR_DOWNLOAD_PATH) / "quarter")
    print(f"  [info ]   latest SEC quarter -> {sec_q}")

    print("-" * 64)
    if missing:
        print(f"MISSING artifacts: {missing}")
    if stale:
        print(f"STALE artifacts:   {[n for n, _ in stale]}")
        print("Result: FAIL")
        return 1
    if missing:
        print("Result: FAIL (missing artifacts)")
        return 1
    print("Result: PASS — all checked artifacts are current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
