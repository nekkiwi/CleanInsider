"""
Daily mark-to-market portfolio backtester.

This module builds a GENUINE daily equity curve from real daily price paths,
in contrast to ``portfolio_daily_returns`` in ``training_helpers.py`` which
spread each trade's total realized return SMOOTHLY across its holding days and
thereby annihilated variance (producing an absurd Sharpe ~22 / MaxDD ~0.8%).

Entry/exit rules MIRROR
``src/scrapers/target_scraper/generate_targets.py`` EXACTLY:

  * entry_idx = searchsorted(stock_dates, entry_date, side="left")
  * 7-day tolerance: skip if the first available trade date is > 7 days after
    the requested entry date
  * entry price = Close[entry_idx]; the lookahead window starts at entry_idx + 1
  * TP fires when High >= entry * (1 + tp); SL fires when Low <= entry * (1 + sl)
  * if both fire on the same bar, TP wins (tie)
  * otherwise the position exits at the horizon end using that day's Close
  * SPX entry/exit indices via searchsorted(side="right") - 1

The module is dependency-injectable: callers pass an ``ohlcv_loader`` callable
and pre-built ``spx_arrays`` so tests never touch disk.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

# Tolerance (days) between requested entry date and first available trade date,
# matching generate_targets.py.
ENTRY_TOLERANCE_DAYS = 7

# Reference annualization factor for daily returns.
TRADING_DAYS = 252


def _default_ohlcv_loader(
    ticker: str,
    db_path_str: Optional[str] = None,
    required_start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Default loader: the real Stooq/yfinance source from data_loader.

    Resolves the Stooq DB path from config when not supplied, so the default
    loader is self-sufficient (callers/tests may still inject their own).
    """
    from src.scrapers.data_loader import load_ohlcv_with_fallback

    if db_path_str is None:
        from src import config

        db_path_str = str(config.STOOQ_DATABASE_PATH)

    return load_ohlcv_with_fallback(
        ticker, db_path_str, required_start_date=required_start_date
    )


def simulate_position_daily_alpha(
    stock_prices: Optional[pd.DataFrame],
    spx_arrays: tuple[np.ndarray, np.ndarray],
    entry_date: pd.Timestamp,
    tp: float,
    sl: float,
    horizon_days: int,
) -> pd.Series:
    """
    Daily alpha return series for ONE position.

    Returns a ``pd.Series`` indexed by business date covering entry_idx+1 through
    the exit day. Each value is the stock's close-to-close return minus the SPX
    close-to-close return over the same day. On the exit day:
      * TP/SL hit intrabar  -> stock return is to the tp/sl threshold PRICE
      * horizon end         -> stock return is to that day's Close

    Returns an empty Series when OHLCV is missing/insufficient or the entry is
    not found within ``ENTRY_TOLERANCE_DAYS`` of ``entry_date``.
    """
    empty = pd.Series(dtype=float)

    if stock_prices is None or stock_prices.empty:
        return empty

    entry_date = pd.Timestamp(entry_date)
    entry_np = np.datetime64(entry_date, "ns")

    stock_dates = stock_prices.index.to_numpy(dtype="datetime64[ns]")
    stock_high = stock_prices["High"].to_numpy(dtype=float)
    stock_low = stock_prices["Low"].to_numpy(dtype=float)
    stock_close = stock_prices["Close"].to_numpy(dtype=float)

    spx_dates, spx_close = spx_arrays

    # --- Entry lookup (mirror generate_targets) ---
    entry_idx = np.searchsorted(stock_dates, entry_np, side="left")
    if entry_idx >= len(stock_dates):
        return empty

    actual_trade_date = stock_dates[entry_idx]
    if (actual_trade_date - entry_np) > np.timedelta64(ENTRY_TOLERANCE_DAYS, "D"):
        return empty

    entry_price = stock_close[entry_idx]
    if entry_price <= 0:
        return empty

    lookahead_start = entry_idx + 1
    if lookahead_start >= len(stock_dates):
        return empty

    # SPX entry price.
    spx_entry_idx = np.searchsorted(spx_dates, actual_trade_date, side="right") - 1
    if spx_entry_idx < 0:
        return empty
    spx_entry_price = spx_close[spx_entry_idx]
    if spx_entry_price <= 0:
        return empty

    tp_price = entry_price * (1 + tp)
    sl_price = entry_price * (1 + sl)

    end_idx = min(lookahead_start + horizon_days, len(stock_dates))
    high_w = stock_high[lookahead_start:end_idx]
    low_w = stock_low[lookahead_start:end_idx]
    close_w = stock_close[lookahead_start:end_idx]
    date_w = stock_dates[lookahead_start:end_idx]

    if len(close_w) == 0:
        return empty

    # --- Determine the exit bar offset and the final-day stock price ---
    tp_hits = np.flatnonzero(high_w >= tp_price)
    sl_hits = np.flatnonzero(low_w <= sl_price)

    if tp_hits.size and (not sl_hits.size or tp_hits[0] <= sl_hits[0]):
        exit_offset = int(tp_hits[0])
        final_stock_price = tp_price  # exit to the TP threshold
    elif sl_hits.size:
        exit_offset = int(sl_hits[0])
        final_stock_price = sl_price  # exit to the SL threshold
    else:
        exit_offset = len(close_w) - 1
        final_stock_price = float(close_w[exit_offset])  # horizon: use Close

    # Build the per-day stock close path from entry through the exit day.
    # Intermediate days use actual Close; the FINAL day uses final_stock_price
    # (threshold on a TP/SL hit, Close at horizon).
    stock_path = np.empty(exit_offset + 2, dtype=float)
    stock_path[0] = entry_price
    if exit_offset > 0:
        stock_path[1 : exit_offset + 1] = close_w[:exit_offset]
    stock_path[exit_offset + 1] = final_stock_price

    stock_daily = stock_path[1:] / stock_path[:-1] - 1.0

    # SPX daily path over the same calendar days (entry day + each lookahead day
    # up to the exit day). Match each day to its SPX close via side="right" - 1.
    spx_path = np.empty(exit_offset + 2, dtype=float)
    spx_path[0] = spx_entry_price
    out_dates = date_w[: exit_offset + 1]
    for i, d in enumerate(out_dates):
        j = np.searchsorted(spx_dates, d, side="right") - 1
        spx_path[i + 1] = spx_close[j] if j >= 0 else spx_path[i]
    spx_daily = spx_path[1:] / spx_path[:-1] - 1.0

    alpha_daily = stock_daily - spx_daily
    return pd.Series(alpha_daily, index=pd.DatetimeIndex(out_dates))


def _position_daily_returns(
    stock_prices: Optional[pd.DataFrame],
    spx_arrays: tuple[np.ndarray, np.ndarray],
    entry_date: pd.Timestamp,
    tp: float,
    sl: float,
    horizon_days: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Return both RAW (stock-only) and ALPHA daily return series for one position.

    The raw series is the alpha series with the SPX leg added back, so both are
    guaranteed to share the exact same index and exit logic.
    """
    alpha = simulate_position_daily_alpha(
        stock_prices, spx_arrays, entry_date, tp, sl, horizon_days
    )
    if alpha.empty:
        return alpha, alpha

    # Recompute the stock-only leg over the same index by re-deriving the SPX
    # daily returns and adding them back. Cheaper: recompute the raw path
    # directly here using the same exit decision the alpha call made.
    raw = _raw_position_daily_returns(
        stock_prices, spx_arrays, entry_date, tp, sl, horizon_days, alpha.index
    )
    return raw, alpha


def _raw_position_daily_returns(
    stock_prices: pd.DataFrame,
    spx_arrays: tuple[np.ndarray, np.ndarray],
    entry_date: pd.Timestamp,
    tp: float,
    sl: float,
    horizon_days: int,
    expected_index: pd.DatetimeIndex,
) -> pd.Series:
    """Stock-only daily returns, aligned to ``expected_index``."""
    entry_np = np.datetime64(pd.Timestamp(entry_date), "ns")
    stock_dates = stock_prices.index.to_numpy(dtype="datetime64[ns]")
    stock_high = stock_prices["High"].to_numpy(dtype=float)
    stock_low = stock_prices["Low"].to_numpy(dtype=float)
    stock_close = stock_prices["Close"].to_numpy(dtype=float)

    entry_idx = np.searchsorted(stock_dates, entry_np, side="left")
    entry_price = stock_close[entry_idx]
    lookahead_start = entry_idx + 1
    end_idx = min(lookahead_start + horizon_days, len(stock_dates))
    high_w = stock_high[lookahead_start:end_idx]
    low_w = stock_low[lookahead_start:end_idx]
    close_w = stock_close[lookahead_start:end_idx]

    tp_price = entry_price * (1 + tp)
    sl_price = entry_price * (1 + sl)
    tp_hits = np.flatnonzero(high_w >= tp_price)
    sl_hits = np.flatnonzero(low_w <= sl_price)

    if tp_hits.size and (not sl_hits.size or tp_hits[0] <= sl_hits[0]):
        exit_offset = int(tp_hits[0])
        final_stock_price = tp_price
    elif sl_hits.size:
        exit_offset = int(sl_hits[0])
        final_stock_price = sl_price
    else:
        exit_offset = len(close_w) - 1
        final_stock_price = float(close_w[exit_offset])

    stock_path = np.empty(exit_offset + 2, dtype=float)
    stock_path[0] = entry_price
    if exit_offset > 0:
        stock_path[1 : exit_offset + 1] = close_w[:exit_offset]
    stock_path[exit_offset + 1] = final_stock_price
    raw_daily = stock_path[1:] / stock_path[:-1] - 1.0
    return pd.Series(raw_daily, index=expected_index)


def simulate_daily_portfolio(
    positions: pd.DataFrame,
    tp: float,
    sl: float,
    horizon_days: int,
    ohlcv_loader: Callable[..., pd.DataFrame] = _default_ohlcv_loader,
    spx_arrays: Optional[tuple[np.ndarray, np.ndarray]] = None,
    db_path: Optional[str] = None,
    per_name_cap: Optional[float] = None,
    max_gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Build a daily mark-to-market portfolio return series under a realistic
    CAPPED TARGET-EXPOSURE capital model.

    Parameters
    ----------
    positions : DataFrame with columns ``Ticker``, ``entry_date``, ``weight``
        ``weight`` is a per-name CAPITAL weight (an ABSOLUTE fraction of the
        book, NOT a relative conviction score that gets renormalized). E.g. a
        weight of 0.05 means 5% of the book in that name.
        An OPTIONAL ``entry_cost`` column (one-way transaction cost as a return
        fraction, e.g. 0.5 * spread) is subtracted from that position's FIRST
        active day return (both ``port_raw`` and ``port_alpha``), scaled by the
        position's capital weight. Absent column -> no cost (backward compatible).
    tp, sl, horizon_days : strategy parameters (mirror generate_targets).
    ohlcv_loader : callable(ticker, db_path_str=, required_start_date=) -> frame
        Injectable price source. Defaults to the real data_loader fallback.
    spx_arrays : (dates ndarray, close ndarray) benchmark series.
    db_path : passed through to the loader as ``db_path_str``.
    per_name_cap : per-position capital cap (default ``config.MAX_POSITION_SIZE``
        = 0.05, i.e. 5%). Each open position's weight is clamped to this.
    max_gross_exposure : maximum total invested fraction on any day (default 1.0
        = no leverage). When the day's summed capped weights exceed this, the
        whole day's weights are scaled down by ``max_gross_exposure / exposure``.

    Capital model
    -------------
    For each day:
      * cap each open position's weight at ``per_name_cap``;
      * exposure = sum of the capped open weights;
      * if exposure > ``max_gross_exposure``, scale that day's weights by
        ``max_gross_exposure / exposure`` (so the total is <= max_gross, per-name
        still <= cap, NO leverage);
      * daily portfolio return = SUM(weight_i * daily_ret_i) over the open
        positions. There is NO renormalization: any unfilled exposure is held
        implicitly in cash (0 return), so sparse days carry a genuine cash drag.
    This is applied identically to both ``port_raw`` and ``port_alpha``. On a
    position's first active day its contribution is
    ``weight_i * (daily_ret_i - entry_cost_i)``. Days with no open position are
    flat (0).

    Returns a DataFrame indexed by business day with columns
    ``['port_raw', 'port_alpha', 'n_open']``. Empty DataFrame if no position
    produces any daily return.
    """
    if per_name_cap is None:
        from src import config

        per_name_cap = config.MAX_POSITION_SIZE

    cols = ["port_raw", "port_alpha", "n_open"]
    if positions is None or len(positions) == 0:
        return pd.DataFrame(columns=cols)

    has_entry_cost = "entry_cost" in positions.columns

    # Cache OHLCV per ticker to avoid redundant loads.
    ohlcv_cache: dict[str, pd.DataFrame] = {}

    raw_series: list[pd.Series] = []
    alpha_series: list[pd.Series] = []
    weights: list[float] = []
    entry_costs: list[float] = []

    for _, row in positions.iterrows():
        ticker = row["Ticker"]
        weight = float(row["weight"])
        if weight <= 0:
            continue

        if ticker not in ohlcv_cache:
            ohlcv_cache[ticker] = ohlcv_loader(
                ticker, db_path_str=db_path, required_start_date=None
            )
        prices = ohlcv_cache[ticker]

        raw, alpha = _position_daily_returns(
            prices, spx_arrays, row["entry_date"], tp, sl, horizon_days
        )
        if alpha.empty:
            continue
        raw_series.append(raw)
        alpha_series.append(alpha)
        weights.append(weight)
        cost = 0.0
        if has_entry_cost:
            cost = row["entry_cost"]
            cost = 0.0 if pd.isna(cost) else float(cost)
        entry_costs.append(cost)

    if not alpha_series:
        return pd.DataFrame(columns=cols)

    # Align all positions onto a shared business-day grid.
    raw_mat = pd.concat(raw_series, axis=1)
    alpha_mat = pd.concat(alpha_series, axis=1)
    raw_mat.columns = range(raw_mat.shape[1])
    alpha_mat.columns = range(alpha_mat.shape[1])

    full_index = pd.bdate_range(raw_mat.index.min(), raw_mat.index.max())
    raw_mat = raw_mat.reindex(full_index)
    alpha_mat = alpha_mat.reindex(full_index)

    # Per-name capital weights, capped at ``per_name_cap``.
    w = np.minimum(np.asarray(weights, dtype=float), float(per_name_cap))
    open_mask = alpha_mat.notna().to_numpy()  # True where a position is open
    n_open = open_mask.sum(axis=1)

    # Subtract each position's one-way entry cost from its FIRST active day,
    # in BOTH the raw and alpha legs (cost is a real cash drag, benchmark-neutral).
    # Contribution is weight-scaled downstream, so we subtract from the per-day
    # return here and let the weight multiplier apply uniformly.
    if has_entry_cost and any(c != 0.0 for c in entry_costs):
        for j, cost in enumerate(entry_costs):
            if cost == 0.0:
                continue
            active_rows = np.flatnonzero(open_mask[:, j])
            if active_rows.size == 0:
                continue
            first = active_rows[0]
            raw_mat.iat[first, j] = raw_mat.iat[first, j] - cost
            alpha_mat.iat[first, j] = alpha_mat.iat[first, j] - cost

    # Per-day CAPITAL weights among open positions (absolute, not renormalized).
    # Cap gross exposure at ``max_gross_exposure`` by scaling the whole day down
    # when its summed capped weights exceed the cap; unfilled exposure is cash.
    weight_mat = np.where(open_mask, w[np.newaxis, :], 0.0)
    exposure = weight_mat.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(
            exposure > max_gross_exposure,
            max_gross_exposure / exposure,
            1.0,
        )
    day_w = weight_mat * scale[:, np.newaxis]

    raw_vals = np.nan_to_num(raw_mat.to_numpy(), nan=0.0)
    alpha_vals = np.nan_to_num(alpha_mat.to_numpy(), nan=0.0)

    port_raw = (raw_vals * day_w).sum(axis=1)
    port_alpha = (alpha_vals * day_w).sum(axis=1)

    out = pd.DataFrame(
        {"port_raw": port_raw, "port_alpha": port_alpha, "n_open": n_open},
        index=full_index,
    )
    return out


def compute_portfolio_metrics(daily_returns: pd.Series) -> dict:
    """
    Performance metrics for a daily return series.

    Returns a dict with:
      sharpe, sortino, max_drawdown (positive magnitude), cagr, ann_vol,
      exposure_days (# nonzero days), n_days.

    Guards: std == 0 or len < 2 -> nan for ratio-based metrics. Empty input is
    handled without crashing.
    """
    r = pd.Series(daily_returns, dtype=float).dropna()
    n = len(r)

    result = {
        "sharpe": np.nan,
        "sortino": np.nan,
        "max_drawdown": np.nan,
        "cagr": np.nan,
        "ann_vol": np.nan,
        "exposure_days": int((r != 0).sum()),
        "n_days": int(n),
    }

    if n < 2:
        return result

    mean = r.mean()
    std = r.std(ddof=1)

    if std and not np.isnan(std):
        result["sharpe"] = float(mean / std * np.sqrt(TRADING_DAYS))
        result["ann_vol"] = float(std * np.sqrt(TRADING_DAYS))

    downside = r[r < 0]
    if len(downside) >= 2:
        dd_std = downside.std(ddof=1)
        if dd_std and not np.isnan(dd_std):
            result["sortino"] = float(mean / dd_std * np.sqrt(TRADING_DAYS))

    # Equity curve, drawdown, CAGR.
    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    result["max_drawdown"] = float(-drawdown.min())

    total_return = float(equity.iloc[-1]) - 1.0
    years = n / TRADING_DAYS
    if years > 0 and (1.0 + total_return) > 0:
        result["cagr"] = float((1.0 + total_return) ** (1.0 / years) - 1.0)

    return result
