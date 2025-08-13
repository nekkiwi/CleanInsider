# file: src/scrapers/target_scraper/calculate_master_targets.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yfinance as yf
import warnings

from src import config as project_config
from src.scrapers.feature_scraper.build_event_ohlcv import _build_future_windows_for_ticker


SPX_TICKER_LOCAL = '^spx'
SPX_TICKER_YFINANCE = '^GSPC'


def get_and_update_spx_data(config, min_required_date: pd.Timestamp):
    # This function remains correct and is unchanged.
    print("\n--- Pre-loading and Verifying SPX Market Data ---")
    local_spx_path = Path(config.STOOQ_DATABASE_PATH) / f"{SPX_TICKER_LOCAL}.parquet"
    try:
        spx_local = pd.read_parquet(local_spx_path)
        if spx_local.index.tz is not None:
             spx_local.index = spx_local.index.tz_localize(None)
    except FileNotFoundError:
        spx_local = pd.DataFrame()
    today = pd.Timestamp.now().normalize()
    download_needed = False
    download_start_date = min_required_date
    if spx_local.empty:
        print("-> Local SPX cache not found. A full download is required.")
        download_needed = True
        download_start_date = min_required_date
    else:
        is_history_sufficient = spx_local.index.min() <= min_required_date
        is_data_fresh = spx_local.index.max() >= today - pd.Timedelta(days=3)
        if is_history_sufficient and is_data_fresh:
            print(f"-> Local SPX data is complete and up-to-date (Covers {spx_local.index.min().date()} to {spx_local.index.max().date()}).")
            return spx_local
        elif not is_history_sufficient:
            print(f"-> Local SPX history is incomplete. Cache starts at {spx_local.index.min().date()}, but data is needed from {min_required_date.date()}.")
            download_needed = True
            download_start_date = min_required_date
            spx_local = pd.DataFrame() 
        elif not is_data_fresh:
            print(f"-> Local SPX data is stale. Latest point is {spx_local.index.max().date()}.")
            download_needed = True
            download_start_date = spx_local.index.max() + pd.Timedelta(days=1)
    if download_needed:
        print(f"-> Attempting to fetch/update SPX data from yfinance starting from {download_start_date.date()}...")
        try:
            spx_yf = yf.download(SPX_TICKER_YFINANCE, start=download_start_date, progress=False, auto_adjust=False)
            if not spx_yf.empty:
                print(f"-> Successfully downloaded {len(spx_yf)} new rows of SPX data.")
                spx_yf.index = spx_yf.index.tz_localize(None)
                updated_spx_data = pd.concat([spx_local, spx_yf])
                updated_spx_data = updated_spx_data[~updated_spx_data.index.duplicated(keep='last')]
                updated_spx_data.sort_index(inplace=True)
                local_spx_path.parent.mkdir(parents=True, exist_ok=True)
                updated_spx_data.to_parquet(local_spx_path)
                return updated_spx_data
            else:
                if spx_local.empty: return pd.DataFrame()
                else:
                    warnings.warn("Could not fetch fresh SPX data; proceeding with existing local data.")
                    return spx_local
        except Exception as e:
            print(f"-> Yfinance download for SPX failed: {e}")
            if spx_local.empty: return pd.DataFrame()
            else:
                warnings.warn("Yfinance download failed; proceeding with existing local data.")
                return spx_local
    return spx_local


def _cs_two_day_series(df: pd.DataFrame) -> pd.Series:
    """
    Corwin–Schultz daily estimator using two consecutive days (t-1, t),
    available at end-of-day t. Requires columns High/Low.
    Returns a Series indexed like df, NaN for the first day.
    """
    if df is None or df.empty or 'High' not in df.columns or 'Low' not in df.columns:
        return pd.Series(index=df.index if df is not None else pd.Index([]), dtype=float)

    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    # Shifted series for t-1
    high_prev = high.shift(1)
    low_prev = low.shift(1)

    # Beta_t over (t-1, t)
    log_hl_prev = np.log(high_prev / low_prev)
    log_hl_curr = np.log(high / low)
    beta_t = (log_hl_prev ** 2) + (log_hl_curr ** 2)

    # Gamma_t over two-day max/min
    max_high = pd.concat([high_prev, high], axis=1).max(axis=1)
    min_low = pd.concat([low_prev, low], axis=1).min(axis=1)
    gamma_t = np.log(max_high / min_low) ** 2

    # Alpha and spread
    with np.errstate(invalid='ignore'):
        alpha = ((np.sqrt(2 * beta_t) - np.sqrt(beta_t)) ** 2) / (3 - 2 * np.sqrt(2))
        alpha = alpha.clip(lower=0)
        spread = 2 * (np.exp(np.sqrt(alpha)) - 1) / (1 + np.exp(np.sqrt(alpha)))

    spread.name = 'cs_spread'
    return spread


# --- THIS IS THE NEW WORKER FUNCTION FOR PARALLEL DATA LOADING ---
def _compute_event_alpha_with_spreads(
    event_window: pd.DataFrame,
    past_hist: pd.DataFrame,
    spx_df: pd.DataFrame,
    tp: float,
    sl: float,
    lookahead_days: int,
) -> float:
    """
    Compute net-of-spread event return and subtract SPX net-of-spread return to produce alpha.
    Assumes event_window contains consecutive rows from entry day (row 0) forward with columns
    ['Date','Open','High','Low','Close','Volume'].
    """
    if event_window is None or event_window.empty:
        return np.nan

    window = event_window.sort_values('Date').reset_index(drop=True)
    # Entry row
    entry_date = pd.to_datetime(window.loc[0, 'Date'])
    entry_close = float(window.loc[0, 'Close'])
    if not np.isfinite(entry_close) or entry_close <= 0:
        return np.nan

    # Entry spread from past history (needs day t-1 and t)
    past_hist = past_hist.copy()
    past_hist = past_hist.sort_values('Date')
    past_hist_idx = past_hist.set_index('Date')
    cs_entry_series = _cs_two_day_series(past_hist_idx[['High','Low']])
    entry_spread = cs_entry_series.reindex([entry_date]).iloc[0] if entry_date in cs_entry_series.index else np.nan
    if not np.isfinite(entry_spread):
        entry_spread = 0.0005  # default 5 bps fallback
    entry_net = entry_close * (1.0 + entry_spread / 2.0)

    # Pre-compute CS for future window days using two-day method within the window (needs t-1)
    fut_idx = window.set_index('Date')
    cs_future = _cs_two_day_series(fut_idx[['High','Low']])

    # Determine exit
    tp_price = entry_net * (1.0 + tp)
    sl_price = entry_net * (1.0 + sl)
    exit_row = None
    n = len(window)
    # Limit to the requested lookahead horizon within the fixed 6-month window
    last_considered = min(n - 1, lookahead_days)
    for i in range(1, last_considered + 1):
        date_i = pd.to_datetime(window.loc[i, 'Date'])
        cs_exit = cs_future.loc[date_i] if date_i in cs_future.index else np.nan
        if not np.isfinite(cs_exit):
            cs_exit = 0.0005
        high_net = float(window.loc[i, 'High']) * (1.0 - cs_exit / 2.0)
        low_net = float(window.loc[i, 'Low']) * (1.0 - cs_exit / 2.0)
        if high_net >= tp_price:
            exit_row = (i, 'TP', cs_exit)
            break
        if low_net <= sl_price:
            exit_row = (i, 'SL', cs_exit)
            break

    if exit_row is None:
        # Exit on the last considered day close
        i = last_considered
        date_i = pd.to_datetime(window.loc[i, 'Date'])
        cs_exit = cs_future.loc[date_i] if date_i in cs_future.index else np.nan
        if not np.isfinite(cs_exit):
            cs_exit = 0.0005
        exit_price = float(window.loc[i, 'Close']) * (1.0 - cs_exit / 2.0)
    else:
        i, reason, cs_exit = exit_row
        if reason == 'TP':
            exit_price = float(window.loc[i, 'High']) * (1.0 - cs_exit / 2.0)
        else:
            exit_price = float(window.loc[i, 'Low']) * (1.0 - cs_exit / 2.0)

    stock_return_net = (exit_price / entry_net) - 1.0

    # SPX net-of-spread return between entry_date and window[i]['Date']
    spx = spx_df.copy()
    spx.index = pd.to_datetime(spx.index)
    spx = spx.sort_index()
    spx_cs = _cs_two_day_series(spx[['High','Low']])

    # Entry SPX index (on or before entry date)
    spx_dates = spx.index.to_numpy(dtype='datetime64[ns]')
    entry_idx = np.searchsorted(spx_dates, entry_date.to_datetime64(), side='right') - 1
    if entry_idx < 1:
        return np.nan
    exit_date = pd.to_datetime(window.loc[i, 'Date'])
    exit_idx = np.searchsorted(spx_dates, exit_date.to_datetime64(), side='right') - 1
    if exit_idx < 1:
        return np.nan

    spx_entry_close = float(spx['Close'].iloc[int(entry_idx)])
    spx_exit_close = float(spx['Close'].iloc[int(exit_idx)])
    entry_spread_val = spx_cs.iloc[int(entry_idx)]
    exit_spread_val = spx_cs.iloc[int(exit_idx)]
    try:
        entry_spread_float = float(entry_spread_val)
    except Exception:
        entry_spread_float = np.nan
    try:
        exit_spread_float = float(exit_spread_val)
    except Exception:
        exit_spread_float = np.nan
    spx_entry_spread = entry_spread_float if (pd.notna(entry_spread_float) and np.isfinite(entry_spread_float)) else 0.0005
    spx_exit_spread = exit_spread_float if (pd.notna(exit_spread_float) and np.isfinite(exit_spread_float)) else 0.0005
    spx_entry_net = spx_entry_close * (1.0 + spx_entry_spread / 2.0)
    spx_exit_net = spx_exit_close * (1.0 - spx_exit_spread / 2.0)
    spx_return_net = (spx_exit_net / spx_entry_net) - 1.0

    return float(stock_return_net - spx_return_net)


def calculate_master_targets(config, target_combinations: list, batch_size: int = 100, debug: bool = False):
    print("\n--- STEP 2: Calculating Master Targets in Batches ---")
    targets_dir = Path(config.TARGETS_OUTPUT_PATH)
    event_list_path = targets_dir / "master_event_list.parquet"
    output_path = targets_dir / "master_targets.parquet"
    if not event_list_path.exists():
        raise FileNotFoundError(f"Master event list not found at {event_list_path}.")
    
    base_df = pd.read_parquet(event_list_path)
    base_df['Filing Date'] = pd.to_datetime(base_df['Filing Date']).dt.tz_localize(None)
    all_tickers = base_df['Ticker'].unique()
    
    if base_df.empty:
        raise ValueError("Master event list is empty.")
    
    min_date_needed = base_df['Filing Date'].min() - pd.Timedelta(days=400)
    spx_data = get_and_update_spx_data(config, min_date_needed)
    
    if spx_data.empty:
        raise RuntimeError("FATAL: Could not load SPX data from any source.")
    if spx_data.index.min() > min_date_needed:
        raise RuntimeError(f"FATAL: Final SPX data does not cover the required historical range.")
        
    print(f"--- ✅ SPX data ready (Range: {spx_data.index.min().date()} to {spx_data.index.max().date()}) ---\n")

    # Preload component parquet paths
    ohlcv_past_path = Path(project_config.OHLCV_PAST_COMPONENT_PATH)
    ohlcv_future_path = Path(project_config.OHLCV_FUTURE_COMPONENT_PATH)
    if not ohlcv_future_path.exists():
        raise FileNotFoundError(f"Future OHLCV component not found at {ohlcv_future_path}")
    if not ohlcv_past_path.exists():
        raise FileNotFoundError(f"Past OHLCV component not found at {ohlcv_past_path}")

    all_results = []
    # Main batch loop should persist on screen
    for i in tqdm(range(0, len(all_tickers), batch_size), desc="Processing Ticker Batches", leave=True):
        ticker_batch = all_tickers[i:i + batch_size]
        batch_df = base_df[base_df['Ticker'].isin(ticker_batch)].copy()

        # Sub-step: Read only needed rows from future and past components
        with tqdm(total=2, desc="Load OHLCV (future/past)", leave=False) as pbar_load:
            try:
                future_batch = pd.read_parquet(
                    ohlcv_future_path, filters=[[('Ticker', 'in', list(ticker_batch))]]
                )
            except Exception:
                future_full = pd.read_parquet(ohlcv_future_path)
                future_batch = future_full[future_full['Ticker'].isin(ticker_batch)]
            pbar_load.update(1)

            try:
                past_batch = pd.read_parquet(
                    ohlcv_past_path, filters=[[('Ticker', 'in', list(ticker_batch))]]
                )
            except Exception:
                past_full = pd.read_parquet(ohlcv_past_path)
                past_batch = past_full[past_full['Ticker'].isin(ticker_batch)]
            pbar_load.update(1)

        # Ensure datetime types
        future_batch['Date'] = pd.to_datetime(future_batch['Date'])
        future_batch['Filing Date'] = pd.to_datetime(future_batch['Filing Date'])
        past_batch['Date'] = pd.to_datetime(past_batch['Date'])

        # Build mapping for quick access to each event window
        future_grouped = future_batch.groupby(['Ticker', 'Filing Date'])

        batch_results_df = batch_df[['Ticker', 'Filing Date']].copy()
        # Sub-step: iterate target combinations (non-persistent)
        for params in tqdm(target_combinations, desc="Target combos", leave=False):
            timepoint, tp, sl = params['time'], params['tp'], params['sl']
            # timepoint ignored now because window is fixed at 6 months; keep naming for compatibility
            col_name = f"alpha_{timepoint}_tp{str(tp).replace('.', 'p')}_sl{str(sl).replace('.', 'p')}"

            alpha_values = []
            # Sub-step: per-event loop (non-persistent)
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Events for {col_name}", leave=False):
                tkr = row['Ticker']
                fdate = row['Filing Date']
                # Future window for event
                try:
                    ev_window = future_grouped.get_group((tkr, fdate))
                except KeyError:
                    alpha_values.append(np.nan)
                    continue
                # Past history for ticker
                # Slice past history to [fdate - 1y, fdate] to avoid unnecessary data
                past_hist_all = past_batch[past_batch['Ticker'] == tkr][['Date','High','Low','Close','Volume']]
                if not past_hist_all.empty:
                    past_hist_all = past_hist_all.sort_values('Date')
                    left = pd.to_datetime(fdate) - pd.Timedelta(days=365)
                    right = pd.to_datetime(fdate)
                    mask = (past_hist_all['Date'] >= left) & (past_hist_all['Date'] <= right)
                    past_hist = past_hist_all.loc[mask]
                else:
                    past_hist = past_hist_all
                # Derive lookahead trading days from the timepoint string (e.g., '1w','2m','10d')
                tp_str = str(timepoint).lower()
                try:
                    if tp_str.endswith('m'):
                        num = int(tp_str[:-1])
                        lookahead_days = max(1, num * 21)
                    elif tp_str.endswith('w'):
                        num = int(tp_str[:-1])
                        lookahead_days = max(1, num * 5)
                    elif tp_str.endswith('d'):
                        num = int(tp_str[:-1])
                        lookahead_days = max(1, num)
                    else:
                        lookahead_days = 21
                except Exception:
                    lookahead_days = 21
                alpha = _compute_event_alpha_with_spreads(ev_window, past_hist, spx_data, tp, sl, lookahead_days)
                alpha_values.append(alpha)

            batch_results_df[col_name] = pd.Series(alpha_values, index=batch_df.index)

        all_results.append(batch_results_df)

    if all_results:
        print("\nCombining results from all batches...")
        final_df = pd.concat(all_results, ignore_index=True)
        print(f"Saving final master targets file to {output_path}")
        final_df.to_parquet(output_path, engine='pyarrow', index=False)
        print("✅ Master target calculations complete.")
    else:
        print("No results were generated to save.")

