import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def parse_timepoint_to_days(tp_str: str) -> int:
    """Convert strings like '3w', '10d', or '2m' to business days."""
    if not isinstance(tp_str, str) or len(tp_str) < 2:
        raise ValueError(f"Invalid timepoint format: {tp_str}")
    unit = tp_str[-1].lower()
    num = int(tp_str[:-1])
    if unit == "d":
        return num
    if unit == "w":
        return num * 5
    if unit == "m":
        return num * 21
    raise ValueError(f"Invalid unit: {unit}")


def _calculate_alpha_for_ticker_group(
    group: pd.DataFrame,
    ticker: str,
    ohlcv_data: dict,
    spx_arrays: tuple[np.ndarray, np.ndarray],
    lookahead_days: int,
    take_profit: float,
    stop_loss: float,
) -> pd.Series:
    """Compute alpha targets for a single ticker group using vectorised lookups.

    The previous implementation relied heavily on pandas slicing and ``asof``
    lookups inside a Python loop which is very slow for large datasets.  This
    version converts the price data to NumPy arrays and uses ``searchsorted``
    to locate the relevant windows, dramatically reducing Python overhead.
    """

    stock_prices = ohlcv_data.get(ticker)
    if stock_prices is None:
        return pd.Series(np.nan, index=group.index)

    # Convert the stock data once to NumPy arrays for fast slicing
    stock_dates = stock_prices.index.to_numpy()
    stock_high = stock_prices["High"].to_numpy()
    stock_low = stock_prices["Low"].to_numpy()
    stock_close = stock_prices["Close"].to_numpy()

    spx_dates, spx_close = spx_arrays

    results = pd.Series(np.nan, index=group.index)

    filing_dates = group["Filing Date"].to_numpy("datetime64[ns]")
    prices = group["Price"].to_numpy()

    for idx, entry_date, entry_price_stock in zip(group.index, filing_dates, prices):
        if (
            np.isnat(entry_date)
            or np.isnan(entry_price_stock)
            or entry_price_stock <= 0
        ):
            continue

        # Locate starting index in the OHLCV arrays
        start_idx = np.searchsorted(stock_dates, entry_date)
        if start_idx >= len(stock_dates):
            continue

        end_idx = min(start_idx + lookahead_days, len(stock_dates) - 1)

        tp_price = entry_price_stock * (1 + take_profit)
        sl_price = entry_price_stock * (1 + stop_loss)

        high_slice = stock_high[start_idx : end_idx + 1]
        low_slice = stock_low[start_idx : end_idx + 1]

        tp_hits = np.flatnonzero(high_slice >= tp_price)
        sl_hits = np.flatnonzero(low_slice <= sl_price)

        if tp_hits.size and (not sl_hits.size or tp_hits[0] <= sl_hits[0]):
            event_idx = start_idx + tp_hits[0]
            event_return = take_profit
        elif sl_hits.size:
            event_idx = start_idx + sl_hits[0]
            event_return = stop_loss
        else:
            event_idx = end_idx
            final_price = stock_close[event_idx]
            event_return = (final_price / entry_price_stock) - 1

        event_date = stock_dates[event_idx]

        # Compute SPX return using searchsorted to mimic ``asof`` behaviour
        spx_start_idx = np.searchsorted(spx_dates, entry_date, side="right") - 1
        spx_end_idx = np.searchsorted(spx_dates, event_date, side="right") - 1
        if spx_start_idx < 0 or spx_end_idx < 0:
            continue

        entry_price_spx = spx_close[spx_start_idx]
        exit_price_spx = spx_close[spx_end_idx]
        if entry_price_spx <= 0:
            continue

        spx_return = (exit_price_spx / entry_price_spx) - 1
        results.loc[idx] = event_return - spx_return

    return results


def calculate_realized_alpha_series(
    base_df: pd.DataFrame,
    ohlcv_data: dict,
    spx_data: pd.DataFrame,
    timepoint_str: str,
    take_profit: float,
    stop_loss: float,
) -> pd.Series:
    """Calculate the continuous alpha target in parallel over ticker groups.

    The S&P 500 data are converted to NumPy arrays once and passed to each
    worker to avoid repeatedly constructing pandas objects.
    """

    lookahead_days = parse_timepoint_to_days(timepoint_str)

    spx_dates = spx_data.index.to_numpy()
    spx_close = spx_data["Close"].to_numpy()
    spx_arrays = (spx_dates, spx_close)

    grouped = base_df.groupby("Ticker")
    tasks = [
        delayed(_calculate_alpha_for_ticker_group)(
            group,
            ticker,
            ohlcv_data,
            spx_arrays,
            lookahead_days,
            take_profit,
            stop_loss,
        )
        for ticker, group in grouped
    ]

    desc = f"Calculating Alpha ({timepoint_str}, TP:{take_profit}, SL:{stop_loss})"
    results_list = Parallel(n_jobs=-2)(tqdm(tasks, desc=desc, total=len(tasks)))
    return pd.concat(results_list)
