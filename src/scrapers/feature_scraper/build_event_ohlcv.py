import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Iterable

from ..data_loader import load_ohlcv_with_fallback


def _load_ticker_history_for_events(
    ticker: str,
    filing_dates: list[pd.Timestamp],
    db_path_str: str,
    past_lookback_calendar_days: int,
) -> tuple[str, pd.DataFrame]:
    """
    Loads OHLCV for a ticker with enough pre-history to cover the largest
    indicator lookback for all its events.
    Returns (ticker, ohlcv_df) where index is DatetimeIndex and columns are
    ['Open','High','Low','Close','Volume'].
    """
    if not filing_dates:
        return ticker, pd.DataFrame()

    min_event_date: pd.Timestamp = min(filing_dates)
    required_start = pd.to_datetime(min_event_date) - pd.Timedelta(days=past_lookback_calendar_days)
    df = load_ohlcv_with_fallback(ticker, db_path_str, required_start_date=required_start)
    return ticker, df


def _build_future_windows_for_ticker(
    ticker: str,
    events_for_ticker: pd.DataFrame,
    full_history: pd.DataFrame,
    future_lookahead_trading_days: int,
) -> pd.DataFrame:
    """
    For a single ticker, slice daily OHLCV windows of exactly
    `future_lookahead_trading_days` after each filing date (inclusive of entry day).
    Drops events that do not have the full window.

    Returns a long-frame with columns:
      ['Ticker','Filing Date','Date','Open','High','Low','Close','Volume']
    """
    if full_history.empty:
        return pd.DataFrame()

    out_rows: list[pd.DataFrame] = []

    # Ensure index and columns
    full_history = full_history.sort_index()
    for _, ev in events_for_ticker.iterrows():
        filing_date: pd.Timestamp = pd.to_datetime(ev['Filing Date'])
        # Find the first trading day on or after filing date
        dates = full_history.index.to_numpy(dtype='datetime64[ns]')
        if dates.size == 0:
            continue
        entry_idx = np.searchsorted(dates, filing_date.to_datetime64(), side='left')
        if entry_idx >= len(dates):
            # No trading day on or after filing date
            continue

        end_idx_exclusive = entry_idx + future_lookahead_trading_days
        if end_idx_exclusive > len(dates):
            # Incomplete future window -> drop event
            continue

        window_df = full_history.iloc[entry_idx:end_idx_exclusive].copy()
        # Attach identifiers
        window_df = window_df.reset_index().rename(columns={'Date': 'Date'})
        window_df.insert(0, 'Filing Date', filing_date.normalize())
        window_df.insert(0, 'Ticker', ticker)
        out_rows.append(window_df[['Ticker', 'Filing Date', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']])

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def build_event_ohlcv_datasets(
    base_df: pd.DataFrame,
    db_path_str: str,
    past_output_path: Path,
    future_output_path: Path,
    past_lookback_calendar_days: int = 400,
    future_lookahead_trading_days: int = 126,
    n_jobs: int = -2,
) -> tuple[Path, Path]:
    """
    Creates two parquet files under features/components/:
      - ohlcv_past.parquet: per-ticker daily OHLCV history sufficient for TA (up to latest event date)
      - ohlcv_future.parquet: per-event daily OHLCV windows for target calculation (exactly 6 months/126 trading days)

    Events that do not have the full 6-month future window are dropped from the
    future parquet by construction.
    """
    if base_df.empty:
        raise ValueError("Base OpenInsider DataFrame is empty")

    # Identify columns
    ticker_col = next((c for c in base_df.columns if c.lower() in ['ticker', 'symbol']), None)
    filing_col = next((c for c in base_df.columns if 'filing' in c.lower() and 'date' in c.lower()), None)
    if ticker_col is None or filing_col is None:
        raise KeyError("Required columns 'Ticker' and 'Filing Date' not found in base_df")

    work_map: dict[str, list[pd.Timestamp]] = (
        base_df[[ticker_col, filing_col]]
        .assign(**{filing_col: pd.to_datetime(base_df[filing_col])})
        .groupby(ticker_col)[filing_col]
        .apply(lambda s: list(s.sort_values().unique()))
        .to_dict()
    )

    # Load all tickers with enough pre-history in parallel
    tasks = [
        delayed(_load_ticker_history_for_events)(
            tkr, dates, db_path_str, past_lookback_calendar_days
        )
        for tkr, dates in work_map.items()
    ]
    results: list[tuple[str, pd.DataFrame]] = Parallel(n_jobs=n_jobs)(
        tqdm(tasks, total=len(tasks), desc="Loading OHLCV for tickers")
    )

    # Build the per-ticker past dataset (history up to the latest event date per ticker)
    past_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []

    for ticker, hist in results:
        if hist is None or hist.empty:
            continue

        hist = hist.copy()
        hist.index.name = 'Date'
        # Cap past history to latest event date for the ticker to avoid forward leakage in 'past'
        latest_event_date = max(work_map[ticker])
        hist_past = hist.loc[:pd.to_datetime(latest_event_date)]
        if not hist_past.empty:
            tmp = hist_past.reset_index()
            tmp.insert(0, 'Ticker', ticker)
            past_frames.append(tmp[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']])

        # Build future windows per event for this ticker
        events_for_ticker = (
            base_df.loc[base_df[ticker_col] == ticker, [ticker_col, filing_col]]
            .rename(columns={ticker_col: 'Ticker', filing_col: 'Filing Date'})
            .drop_duplicates()
        )
        fut_df = _build_future_windows_for_ticker(
            ticker=ticker,
            events_for_ticker=events_for_ticker,
            full_history=hist,
            future_lookahead_trading_days=future_lookahead_trading_days,
        )
        if not fut_df.empty:
            future_frames.append(fut_df)

    # Concatenate and save
    past_all = pd.concat(past_frames, ignore_index=True) if past_frames else pd.DataFrame(
        columns=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    # Drop potential duplicates and sort for efficient reads later
    if not past_all.empty:
        past_all.drop_duplicates(subset=['Ticker', 'Date'], inplace=True)
        past_all.sort_values(['Ticker', 'Date'], inplace=True)
    past_output_path.parent.mkdir(parents=True, exist_ok=True)
    past_all.to_parquet(past_output_path, index=False)

    future_all = pd.concat(future_frames, ignore_index=True) if future_frames else pd.DataFrame(
        columns=['Ticker', 'Filing Date', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    # Also ensure ordering
    if not future_all.empty:
        future_all.sort_values(['Ticker', 'Filing Date', 'Date'], inplace=True)
    future_output_path.parent.mkdir(parents=True, exist_ok=True)
    future_all.to_parquet(future_output_path, index=False)

    return past_output_path, future_output_path







