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


def _calculate_alpha_for_ticker_group_multi(
    group: pd.DataFrame,
    ticker: str,
    stock_prices: pd.DataFrame,
    spx_arrays: tuple[np.ndarray, np.ndarray],
    combos: list[dict],
    debug: bool = False,
) -> tuple[dict[str, pd.Series], list]:
    """
    Compute alpha targets for ALL combos in a SINGLE pass per event.

    For each event we:
    1. Find entry_idx / entry_price once.
    2. Extract the forward window up to the MAX horizon once.
    3. For each combo, apply TP/SL threshold checks against the already-sliced
       arrays using only that combo's horizon h, then look up the SPX exit.

    Returns a dict mapping col_name -> pd.Series (aligned to group.index) and
    a list of debug messages.
    """
    debug_messages = []
    spx_dates, spx_close = spx_arrays

    # Build empty result series for each combo keyed by column name.
    col_names = [c["col_name"] for c in combos]
    empty = {cn: pd.Series(np.nan, index=group.index) for cn in col_names}

    if stock_prices is None or stock_prices.empty:
        return empty, debug_messages

    stock_dates = stock_prices.index.to_numpy(dtype="datetime64[ns]")
    stock_high = stock_prices["High"].to_numpy()
    stock_low = stock_prices["Low"].to_numpy()
    stock_close = stock_prices["Close"].to_numpy()

    # Pre-compute the max horizon across all combos — used to slice the
    # forward window once per event.
    max_horizon = max(c["lookahead_days"] for c in combos)

    results = {cn: pd.Series(np.nan, index=group.index) for cn in col_names}
    filing_dates = group["Filing Date"].to_numpy("datetime64[ns]")

    for idx, filing_date in zip(group.index, filing_dates):
        if np.isnat(filing_date):
            continue

        # --- Entry lookup (once per event) ---
        entry_idx = np.searchsorted(stock_dates, filing_date, side="left")

        if entry_idx >= len(stock_dates):
            if debug:
                debug_messages.append(
                    f"Ticker {ticker}, Filing {np.datetime_as_string(filing_date, 'D')}: "
                    "No stock data found after filing date."
                )
            continue

        actual_trade_date = stock_dates[entry_idx]

        # 7-day tolerance: skip events where historical data is missing.
        if (actual_trade_date - filing_date) > np.timedelta64(7, "D"):
            if debug:
                debug_messages.append(
                    f"Ticker {ticker}, Filing {np.datetime_as_string(filing_date, 'D')}: "
                    f"Historical data missing. First available trade date is "
                    f"{np.datetime_as_string(actual_trade_date, 'D')}."
                )
            continue

        actual_entry_price = stock_close[entry_idx]
        if actual_entry_price <= 0:
            continue

        lookahead_start_idx = entry_idx + 1
        if lookahead_start_idx >= len(stock_dates):
            continue

        # SPX entry index (once per event) — same across all combos.
        spx_entry_idx = np.searchsorted(spx_dates, actual_trade_date, side="right") - 1
        if spx_entry_idx < 0:
            continue
        entry_price_spx = spx_close[spx_entry_idx]
        if entry_price_spx <= 0:
            continue

        # Extract the widest forward window needed by any combo (once).
        max_end_idx = min(lookahead_start_idx + max_horizon, len(stock_dates))
        high_window = stock_high[lookahead_start_idx:max_end_idx]
        low_window = stock_low[lookahead_start_idx:max_end_idx]
        close_window = stock_close[lookahead_start_idx:max_end_idx]
        date_window = stock_dates[lookahead_start_idx:max_end_idx]

        # --- Per-combo threshold checks (cheap — no new data loading) ---
        for combo in combos:
            h = combo["lookahead_days"]
            tp = combo["take_profit"]
            sl = combo["stop_loss"]
            cn = combo["col_name"]

            tp_price = actual_entry_price * (1 + tp)
            sl_price = actual_entry_price * (1 + sl)

            # Clip slice to this combo's horizon.
            high_slice = high_window[:h]
            low_slice = low_window[:h]

            tp_hits = np.flatnonzero(high_slice >= tp_price)
            sl_hits = np.flatnonzero(low_slice <= sl_price)

            if tp_hits.size and (not sl_hits.size or tp_hits[0] <= sl_hits[0]):
                # Take-profit fires first.
                exit_offset = tp_hits[0]
                event_return = tp
            elif sl_hits.size:
                # Stop-loss fires first.
                exit_offset = sl_hits[0]
                event_return = sl
            else:
                # Horizon end — use actual close price.
                exit_offset = min(h, len(close_window)) - 1
                if exit_offset < 0:
                    continue
                event_return = float(close_window[exit_offset] / actual_entry_price) - 1

            exit_date = date_window[exit_offset]
            if np.isnat(exit_date):
                continue

            # SPX exit lookup for this combo's exit date.
            spx_exit_idx = np.searchsorted(spx_dates, exit_date, side="right") - 1
            if spx_exit_idx < 0:
                continue

            exit_price_spx = spx_close[spx_exit_idx]
            spx_return = float(exit_price_spx / entry_price_spx) - 1

            results[cn].loc[idx] = event_return - spx_return

            if debug:
                debug_messages.append(
                    f"Ticker {ticker} [{cn}], Entry "
                    f"{np.datetime_as_string(actual_trade_date, 'D')}, Exit "
                    f"{np.datetime_as_string(exit_date, 'D')}, "
                    f"Stock Ret {event_return:.4f}, SPX Ret {spx_return:.4f}, "
                    f"Alpha {results[cn].loc[idx]:.4f}"
                )

    return results, debug_messages


def calculate_all_alpha_series(
    base_df: pd.DataFrame,
    ohlcv_data: dict,
    spx_data: pd.DataFrame,
    target_combinations: list[dict],
    debug: bool = False,
) -> tuple[pd.DataFrame, list]:
    """
    Compute alpha for ALL combos in a single parallel pass (one groupby).

    Each worker processes one ticker and returns alpha values for all 12
    combos simultaneously, so the expensive data-loading overhead is paid
    only once per ticker rather than once per combo per ticker.

    Returns
    -------
    results_df : pd.DataFrame
        Indexed like base_df, with one alpha_* column per combo.
    all_debug_messages : list[str]
    """
    spx_arrays = (
        spx_data.index.to_numpy(dtype="datetime64[ns]"),
        spx_data["Close"].to_numpy(),
    )

    # Enrich each combo dict with pre-computed lookahead_days and col_name so
    # workers never need to recompute them.
    combos: list[dict] = []
    for params in target_combinations:
        timepoint = params["time"]
        tp = params["tp"]
        sl = params["sl"]
        col_name = (
            f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}"
            f"_sl{str(sl).replace('.', 'p')}"
        )
        combos.append(
            {
                "col_name": col_name,
                "lookahead_days": parse_timepoint_to_days(timepoint),
                "take_profit": tp,
                "stop_loss": sl,
            }
        )

    grouped = base_df.groupby("Ticker")
    tasks = [
        delayed(_calculate_alpha_for_ticker_group_multi)(
            group,
            ticker,
            ohlcv_data.get(ticker),
            spx_arrays,
            combos,
            debug,
        )
        for ticker, group in grouped
    ]

    parallel_results = Parallel(n_jobs=-1)(
        tqdm(
            tasks,
            desc="Calculating all alphas (single pass)",
            total=len(tasks),
            leave=False,
        )
    )

    # --- Assemble per-column Series, then build the DataFrame. ---
    col_names = [c["col_name"] for c in combos]
    accumulated: dict[str, list[pd.Series]] = {cn: [] for cn in col_names}
    all_debug_messages: list[str] = []

    for res_dict, dbg_msgs in parallel_results:
        if dbg_msgs:
            all_debug_messages.extend(dbg_msgs)
        for cn in col_names:
            s = res_dict.get(cn)
            if s is not None and not s.empty:
                accumulated[cn].append(s)

    # Build result DataFrame aligned to base_df.index.
    results_df = pd.DataFrame(index=base_df.index)
    for cn in col_names:
        parts = accumulated[cn]
        if parts:
            results_df[cn] = pd.concat(parts)
        else:
            results_df[cn] = np.nan

    return results_df, all_debug_messages


# ---------------------------------------------------------------------------
# Legacy single-combo API — kept for backward compatibility / tests.
# ---------------------------------------------------------------------------


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
    """
    Original single-combo worker — preserved verbatim for parity verification.
    """
    stock_prices = ohlcv_data.get(ticker)
    debug_messages = []
    spx_dates, spx_close = spx_arrays

    if stock_prices is None or stock_prices.empty:
        return pd.Series(np.nan, index=group.index), debug_messages

    stock_dates, stock_high, stock_low, stock_close = (
        stock_prices.index.to_numpy(dtype="datetime64[ns]"),
        stock_prices["High"].to_numpy(),
        stock_prices["Low"].to_numpy(),
        stock_prices["Close"].to_numpy(),
    )

    results = pd.Series(np.nan, index=group.index)
    filing_dates = group["Filing Date"].to_numpy("datetime64[ns]")

    for idx, filing_date in zip(group.index, filing_dates):
        if np.isnat(filing_date):
            continue

        entry_idx = np.searchsorted(stock_dates, filing_date, side="left")

        if entry_idx >= len(stock_dates):
            if debug:
                debug_messages.append(
                    f"Ticker {ticker}, Filing {np.datetime_as_string(filing_date, 'D')}: No stock data found after filing date."
                )
            continue

        actual_trade_date = stock_dates[entry_idx]

        if (actual_trade_date - filing_date) > np.timedelta64(7, "D"):
            if debug:
                debug_messages.append(
                    f"Ticker {ticker}, Filing {np.datetime_as_string(filing_date, 'D')}: "
                    f"Historical data missing. First available trade date is {np.datetime_as_string(actual_trade_date, 'D')}."
                )
            continue

        actual_entry_price = stock_close[entry_idx]
        if actual_entry_price <= 0:
            continue

        lookahead_start_idx = entry_idx + 1
        if lookahead_start_idx >= len(stock_dates):
            continue

        lookahead_end_idx = min(lookahead_start_idx + lookahead_days, len(stock_dates))

        tp_price = actual_entry_price * (1 + take_profit)
        sl_price = actual_entry_price * (1 + stop_loss)

        high_slice = stock_high[lookahead_start_idx:lookahead_end_idx]
        low_slice = stock_low[lookahead_start_idx:lookahead_end_idx]

        tp_hits = np.flatnonzero(high_slice >= tp_price)
        sl_hits = np.flatnonzero(low_slice <= sl_price)

        event_idx = -1
        if tp_hits.size and (not sl_hits.size or tp_hits[0] <= sl_hits[0]):
            event_idx = lookahead_start_idx + tp_hits[0]
            event_return = take_profit
        elif sl_hits.size:
            event_idx = lookahead_start_idx + sl_hits[0]
            event_return = stop_loss
        else:
            event_idx = lookahead_end_idx - 1
            final_price = stock_close[event_idx]
            event_return = float((final_price / actual_entry_price) - 1)

        if event_idx < 0 or event_idx >= len(stock_dates):
            continue

        exit_date = stock_dates[event_idx]
        if np.isnat(exit_date):
            continue

        spx_entry_idx = np.searchsorted(spx_dates, actual_trade_date, side="right") - 1
        spx_exit_idx = np.searchsorted(spx_dates, exit_date, side="right") - 1

        if spx_entry_idx < 0 or spx_exit_idx < 0:
            continue

        entry_price_spx = spx_close[spx_entry_idx]
        if entry_price_spx <= 0:
            continue

        exit_price_spx = spx_close[spx_exit_idx]
        spx_return = float((exit_price_spx / entry_price_spx) - 1)

        results.loc[idx] = event_return - spx_return

        if debug:
            debug_messages.append(
                f"Ticker {ticker}, Entry {np.datetime_as_string(actual_trade_date, 'D')}, Exit {np.datetime_as_string(exit_date, 'D')}, "
                f"Stock Ret {event_return:.4f}, SPX Ret {spx_return:.4f}, Alpha {results.loc[idx]:.4f}"
            )

    return results, debug_messages


def calculate_realized_alpha_series(
    base_df: pd.DataFrame,
    ohlcv_data: dict,
    spx_data: pd.DataFrame,
    timepoint_str: str,
    take_profit: float,
    stop_loss: float,
    debug: bool = False,
) -> tuple[pd.Series, list]:
    """Calculate alpha targets and aggregate debug messages from parallel workers."""
    lookahead_days = parse_timepoint_to_days(timepoint_str)
    spx_arrays = (
        spx_data.index.to_numpy(dtype="datetime64[ns]"),
        spx_data["Close"].to_numpy(),
    )

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
    parallel_results = Parallel(n_jobs=-2)(
        tqdm(tasks, desc=desc, total=len(tasks), leave=False)
    )

    results_list = [
        res[0] for res in parallel_results if res is not None and not res[0].empty
    ]
    debug_logs = [res[1] for res in parallel_results if res is not None and res[1]]

    all_debug_messages = [msg for sublist in debug_logs for msg in sublist]

    if not results_list:
        return pd.Series(dtype=float), all_debug_messages

    return pd.concat(results_list), all_debug_messages
