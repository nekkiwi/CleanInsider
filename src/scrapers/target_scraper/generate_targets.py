# file: src/scrapers/target_scraper/generate_targets.py

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from tqdm import tqdm
from joblib import Parallel, delayed

def parse_timepoint_to_days(tp_str: str) -> int:
    """Converts a string like '3w', '10d', or '2m' to an integer number of business days."""
    if not isinstance(tp_str, str) or len(tp_str) < 2:
        raise ValueError(f"Invalid timepoint format: {tp_str}")
    unit, num = tp_str[-1].lower(), int(tp_str[:-1])
    if unit == 'd': return num
    if unit == 'w': return num * 5
    if unit == 'm': return num * 21
    raise ValueError(f"Invalid unit: {unit}")

def _calculate_alpha_for_ticker_group(
    group: pd.DataFrame, ticker: str, ohlcv_data: dict, spx_data: pd.DataFrame,
    lookahead_days: int, take_profit: float, stop_loss: float
) -> pd.Series:
    """
    Performs the target calculation for all rows belonging to a single ticker
    using a more efficient loop and vectorized lookups.
    """
    stock_prices = ohlcv_data.get(ticker)
    if stock_prices is None:
        return pd.Series(np.nan, index=group.index)

    results = pd.Series(np.nan, index=group.index)

    # This loop is more explicit and faster than group.apply()
    for idx, row in group.iterrows():
        entry_date = pd.to_datetime(row['Filing Date'])
        entry_price_stock = row['Price'] # Use the price from the features file
        
        if pd.isna(entry_date) or entry_price_stock <= 0:
            continue

        # 1. Calculate absolute price barriers
        tp_price = entry_price_stock * (1 + take_profit)
        sl_price = entry_price_stock * (1 + stop_loss)

        # 2. Define the lookup window
        end_date = entry_date + BDay(lookahead_days)
        price_window = stock_prices.loc[entry_date:end_date]
        
        if price_window.empty or len(price_window) < 2:
            continue
        
        # 3. Vectorized search for the first touch of each barrier
        tp_hits = price_window['High'] >= tp_price
        sl_hits = price_window['Low'] <= sl_price
        
        # .idxmax() finds the index of the first 'True'. If none, it would pick the first row,
        # so we check '.any()' first to handle cases where a barrier is never hit.
        first_tp_date = tp_hits.idxmax() if tp_hits.any() else pd.NaT
        first_sl_date = sl_hits.idxmax() if sl_hits.any() else pd.NaT

        # 4. Determine which event (if any) happened first
        event_date = None
        event_return = None

        if not pd.isna(first_tp_date) and (pd.isna(first_sl_date) or first_tp_date <= first_sl_date):
            event_date = first_tp_date
            event_return = take_profit
        elif not pd.isna(first_sl_date):
            event_date = first_sl_date
            event_return = stop_loss
        else: # If no barrier was hit, use the end of the window
            event_date = price_window.index[-1]
            final_price = price_window.loc[event_date]['Close']
            event_return = (final_price / entry_price_stock) - 1

        # 5. Calculate the SPX return over the same period for alpha
        # Use .asof for robustness against non-trading days
        entry_date_in_spx = spx_data.index.asof(entry_date)
        event_date_in_spx = spx_data.index.asof(event_date)

        if pd.isna(entry_date_in_spx) or pd.isna(event_date_in_spx):
            continue
            
        entry_price_spx = spx_data.loc[entry_date_in_spx]['Close']
        exit_price_spx = spx_data.loc[event_date_in_spx]['Close']
        
        if entry_price_spx <= 0:
            continue

        spx_return = (exit_price_spx / entry_price_spx) - 1
        final_alpha = event_return - spx_return

        results.loc[idx] = final_alpha
        
    return results

def calculate_realized_alpha_series(
    base_df: pd.DataFrame, ohlcv_data: dict, spx_data: pd.DataFrame,
    timepoint_str: str, take_profit: float, stop_loss: float
) -> pd.Series:
    """Calculates the continuous alpha target by parallelizing over ticker groups."""
    lookahead_days = parse_timepoint_to_days(timepoint_str)
    
    grouped = base_df.groupby('Ticker')
    tasks = [
        delayed(_calculate_alpha_for_ticker_group)(
            group, ticker, ohlcv_data, spx_data, lookahead_days, take_profit, stop_loss
        ) for ticker, group in grouped
    ]
    
    desc = f"Calculating Alpha ({timepoint_str}, TP:{take_profit}, SL:{stop_loss})"
    results_list = Parallel(n_jobs=-2)(tqdm(tasks, desc=desc, total=len(tasks)))
    return pd.concat(results_list)