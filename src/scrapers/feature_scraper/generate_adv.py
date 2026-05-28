# file: src/scrapers/feature_scraper/generate_adv.py
"""
Point-in-time Average Dollar Volume (ADV) generator for the liquid-universe pivot.

For every (Ticker, Filing Date) in the master event list, compute the 60-day
rolling MEDIAN of Close*Volume (min_periods=20), taken AS-OF the filing date
(the last rolling value whose date is <= Filing Date). This is the real
liquidity proxy that replaces the volatility-contaminated Corwin-Schultz spread:
``adv >= config.LIQUID_ADV_MIN`` and ``Price >= config.LIQUID_PRICE_MIN`` define
the tradeable universe.

Output: data/scrapers/features/components/adv.parquet
    columns: Ticker, Filing Date, adv

The point-in-time lookup reuses the per-ticker groupby + ``np.searchsorted(
side="right") - 1`` asof pattern from the target scraper, so there is no
look-ahead: a filing before any bar, or before the 20-observation warmup, yields
NaN (unknown / untradeable liquidity).

Usage::

    python -m src.scrapers.feature_scraper.generate_adv

(reads ``config.MASTER_EVENT_LIST_PATH``, loads OHLCV local-only from the Stooq
DB, and writes ``config.ADV_COMPONENT_PATH``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.scrapers.data_loader import load_ohlcv_with_fallback

ADV_WINDOW = 60  # rolling window (calendar/trading bars) for the dollar-volume median
ADV_MIN_PERIODS = 20  # min observations before an ADV value is defined


def _local_only_loader(ticker: str, db_path_str: str, local_only: bool = True):
    """Default OHLCV loader: local Stooq only (fast, offline, no rate limits)."""
    return load_ohlcv_with_fallback(ticker, db_path_str, local_only=local_only)


def compute_pit_adv_for_ticker(ohlcv_df: pd.DataFrame, filing_dates) -> np.ndarray:
    """Point-in-time ADV for one ticker's filing dates.

    Parameters
    ----------
    ohlcv_df : OHLCV frame indexed by date (must have Close and Volume).
    filing_dates : iterable of filing-date timestamps for this ticker.

    Returns
    -------
    np.ndarray of ADV values aligned to ``filing_dates`` (NaN where no value is
    available on/before the filing date or before the min-periods warmup).
    """
    filing = pd.to_datetime(pd.Series(list(filing_dates))).to_numpy("datetime64[ns]")
    out = np.full(len(filing), np.nan)

    if ohlcv_df is None or ohlcv_df.empty:
        return out
    if "Close" not in ohlcv_df.columns or "Volume" not in ohlcv_df.columns:
        return out

    df = ohlcv_df.sort_index()
    dollar_vol = df["Close"].astype(float) * df["Volume"].astype(float)
    # 60-day rolling MEDIAN (median ignores volume spikes that would inflate a mean).
    adv_series = dollar_vol.rolling(
        window=ADV_WINDOW, min_periods=ADV_MIN_PERIODS
    ).median()

    adv_dates = adv_series.index.to_numpy("datetime64[ns]")
    adv_vals = adv_series.to_numpy(dtype=float)

    for i, fd in enumerate(filing):
        if np.isnat(fd):
            continue
        # asof: last index with date <= filing date (no look-ahead).
        pos = np.searchsorted(adv_dates, fd, side="right") - 1
        if pos < 0:
            continue
        out[i] = adv_vals[pos]
    return out


def generate_adv(
    master_df: Optional[pd.DataFrame] = None,
    master_events_path: Optional[Path] = None,
    db_path_str: Optional[str] = None,
    loader: Callable[..., pd.DataFrame] = _local_only_loader,
    output_path: Optional[Path] = None,
    local_only: bool = True,
) -> pd.DataFrame:
    """Compute point-in-time ADV for every (Ticker, Filing Date) and persist it.

    Parameters
    ----------
    master_df : pre-loaded master events frame (Ticker, Filing Date[, Price]).
        If None, read from ``master_events_path`` / ``config.MASTER_EVENT_LIST_PATH``.
    master_events_path : path to the master event list parquet.
    db_path_str : Stooq DB path string passed to ``loader``.
        Defaults to ``str(config.STOOQ_DATABASE_PATH)``.
    loader : ``loader(ticker, db_path_str, local_only=)`` -> OHLCV DataFrame.
    output_path : where to write the adv parquet.
        Defaults to ``config.ADV_COMPONENT_PATH``.
    local_only : forwarded to the loader (default True: local Stooq only).

    Returns
    -------
    DataFrame with columns ``[Ticker, Filing Date, adv]`` (also written to disk).
    """
    if master_df is None:
        path = Path(master_events_path or config.MASTER_EVENT_LIST_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Master events file not found: {path}")
        master_df = pd.read_parquet(path)

    if db_path_str is None:
        db_path_str = str(config.STOOQ_DATABASE_PATH)
    if output_path is None:
        output_path = config.ADV_COMPONENT_PATH

    df = master_df[["Ticker", "Filing Date"]].copy()
    df["Filing Date"] = pd.to_datetime(df["Filing Date"])
    df["adv"] = np.nan

    groups = list(df.groupby("Ticker"))
    for ticker, group in tqdm(groups, desc="Computing point-in-time ADV"):
        ohlcv = loader(ticker, db_path_str, local_only=local_only)
        adv_vals = compute_pit_adv_for_ticker(ohlcv, group["Filing Date"].to_numpy())
        df.loc[group.index, "adv"] = adv_vals

    result = df[["Ticker", "Filing Date", "adv"]].reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    n_known = int(result["adv"].notna().sum())
    print(
        f"[ADV] Wrote {len(result)} (Ticker, Filing Date) rows to {output_path} "
        f"({n_known} with known ADV, {len(result) - n_known} NaN)."
    )
    return result


def main() -> None:
    generate_adv()


if __name__ == "__main__":
    main()
