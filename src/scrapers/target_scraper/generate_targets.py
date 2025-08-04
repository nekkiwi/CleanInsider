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
    debug: bool = False,
) -> tuple[pd.Series, list]:
    """Compute alpha targets, now with enhanced debugging to report worker state."""

    stock_prices = ohlcv_data.get(ticker)
    debug_messages = []

    # --- THE PROOF: Add SPX state info to the debug log ---
    spx_dates, spx_close = spx_arrays
    if debug:
        spx_info = f"WORKER STATE for Ticker '{ticker}': Received SPX data with {len(spx_dates)} rows, from {np.datetime_as_string(spx_dates[0], 'D')} to {np.datetime_as_string(spx_dates[-1], 'D')}."
        debug_messages.append(spx_info)
    # --- END OF THE PROOF ---

    if stock_prices is None:
        if debug:
            debug_messages.append(f"TICKER '{ticker}': No OHLCV data found.")
        return pd.Series(np.nan, index=group.index), debug_messages

    stock_dates, stock_high, stock_low, stock_close = (
        stock_prices.index.to_numpy(),
        stock_prices["High"].to_numpy(),
        stock_prices["Low"].to_numpy(),
        stock_prices["Close"].to_numpy(),
    )

    results = pd.Series(np.nan, index=group.index)
    filing_dates = group["Filing Date"].to_numpy("datetime64[ns]")
    prices = group["Price"].to_numpy()

    for idx, entry_date, entry_price_stock in zip(group.index, filing_dates, prices):
        if (
            np.isnat(entry_date)
            or np.isnan(entry_price_stock)
            or entry_price_stock <= 0
        ):
            if debug:
                debug_messages.append(
                    f"TICKER '{ticker}' on {np.datetime_as_string(entry_date, unit='D')}: Invalid input (NaN date/price or non-positive price)."
                )
            continue

        start_idx = np.searchsorted(stock_dates, entry_date)
        if start_idx >= len(stock_dates):
            if debug:
                debug_messages.append(
                    f"TICKER '{ticker}' on {np.datetime_as_string(entry_date, unit='D')}: Filing date is after last available stock price date."
                )
            continue

        if start_idx + 1 >= len(stock_dates):
            if debug:
                debug_messages.append(
                    f"TICKER '{ticker}' on {np.datetime_as_string(entry_date, unit='D')}: No future price data available to form a lookahead window."
                )
            continue

        end_idx = min(start_idx + lookahead_days, len(stock_dates) - 1)

        # --- Barrier calculation logic (unchanged) ---
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

        spx_start_idx = np.searchsorted(spx_dates, entry_date, side="right") - 1
        spx_end_idx = np.searchsorted(spx_dates, event_date, side="right") - 1

        if spx_start_idx < 0 or spx_end_idx < 0:
            if debug:
                debug_messages.append(
                    f"TICKER '{ticker}' on {np.datetime_as_string(entry_date, unit='D')}: Date range is outside available SPX history."
                )
            continue

        entry_price_spx = spx_close[spx_start_idx]
        if entry_price_spx <= 0:
            if debug:
                debug_messages.append(
                    f"TICKER '{ticker}' on {np.datetime_as_string(entry_date, unit='D')}: Invalid SPX entry price (<= 0)."
                )
            continue

        exit_price_spx = spx_close[spx_end_idx]
        spx_return = (exit_price_spx / entry_price_spx) - 1
        results.loc[idx] = event_return - spx_return

    return results, debug_messages  # <-- Return both results and the log


def calculate_realized_alpha_series(
    base_df: pd.DataFrame,
    ohlcv_data: dict,
    spx_data: pd.DataFrame,
    timepoint_str: str,
    take_profit: float,
    stop_loss: float,
    debug: bool = False,  # Pass debug flag
) -> tuple[pd.Series, list]:  # <-- Return type is now a tuple
    """Calculate alpha targets and aggregate debug messages from parallel workers."""
    lookahead_days = parse_timepoint_to_days(timepoint_str)
    spx_arrays = (spx_data.index.to_numpy(), spx_data["Close"].to_numpy())

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
            debug,
        )
        for ticker, group in grouped
    ]

    desc = f"Calculating Alpha ({timepoint_str}, TP:{take_profit}, SL:{stop_loss})"
    # The result is now a list of tuples: [(series1, log1), (series2, log2), ...]
    parallel_results = Parallel(n_jobs=-2)(tqdm(tasks, desc=desc, total=len(tasks)))

    # Unpack the results
    results_list = [res[0] for res in parallel_results]
    debug_logs = [
        res[1] for res in parallel_results if res[1]
    ]  # Collect non-empty logs

    # Flatten the list of lists into a single list of messages
    all_debug_messages = [msg for sublist in debug_logs for msg in sublist]

    return pd.concat(results_list), all_debug_messages
