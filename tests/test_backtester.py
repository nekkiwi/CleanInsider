"""
Tests for src/training/backtester.py — a daily mark-to-market portfolio
backtester. All price data is synthetic and injected; no disk access.

The backtester must MIRROR the entry/exit rules in
src/scrapers/target_scraper/generate_targets.py:
  - entry_idx = searchsorted(stock_dates, entry_date, side="left")
  - 7-day tolerance on first available trade date
  - entry price = Close[entry_idx]; lookahead starts at entry_idx + 1
  - TP fires when High >= entry*(1+tp); SL fires when Low <= entry*(1+sl)
  - tie on the same bar => TP wins
  - otherwise exit at horizon end using that day's Close
  - SPX entry/exit via searchsorted(side="right") - 1

Unlike the prior smoothed metric, daily portfolio variance here must come from
real day-to-day price moves.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.backtester import (  # noqa: E402
    compute_portfolio_metrics,
    simulate_daily_portfolio,
    simulate_position_daily_alpha,
)

BDAYS = pd.bdate_range("2024-01-01", periods=40)


def _ohlcv(dates, close, high=None, low=None):
    """Build an OHLCV frame indexed by date. High/Low default to Close."""
    close = np.asarray(close, dtype=float)
    high = close.copy() if high is None else np.asarray(high, dtype=float)
    low = close.copy() if low is None else np.asarray(low, dtype=float)
    return pd.DataFrame(
        {"High": high, "Low": low, "Close": close},
        index=pd.DatetimeIndex(dates),
    )


def _spx_arrays(dates, close):
    return (
        pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[ns]"),
        np.asarray(close, dtype=float),
    )


def _flat_spx(dates, level=100.0):
    return _spx_arrays(dates, np.full(len(dates), level))


# ---------------------------------------------------------------------------
# simulate_position_daily_alpha
# ---------------------------------------------------------------------------


def test_single_position_daily_alpha_matches_close_to_close():
    """Known up-then-down close path, flat SPX, horizon exit (no TP/SL hit)."""
    dates = BDAYS[:6]
    # entry on dates[0] @ 100; then 102, 101, 103, 99, 98
    close = [100.0, 102.0, 101.0, 103.0, 99.0, 98.0]
    prices = _ohlcv(dates, close)
    spx = _flat_spx(dates)

    # tp/sl never hit (high == low == close, +/-50% thresholds), horizon 5
    s = simulate_position_daily_alpha(
        prices, spx, entry_date=dates[0], tp=0.50, sl=-0.50, horizon_days=5
    )

    # Daily series from entry+1 through horizon end (5 lookahead bars).
    expected_dates = list(dates[1:6])
    assert list(s.index) == expected_dates

    # Close-to-close stock returns, SPX flat => alpha == stock return.
    expected = [
        102.0 / 100.0 - 1,
        101.0 / 102.0 - 1,
        103.0 / 101.0 - 1,
        99.0 / 103.0 - 1,
        98.0 / 99.0 - 1,
    ]
    np.testing.assert_allclose(s.values, expected, rtol=1e-12)
    # Compounded daily return equals the single-shot horizon return.
    np.testing.assert_allclose((1 + s.values).prod() - 1, 98.0 / 100.0 - 1, rtol=1e-12)


def test_tp_hit_same_first_day():
    """TP hits on the very first lookahead bar; series has one day to threshold."""
    dates = BDAYS[:6]
    close = [100.0, 104.0, 120.0, 120.0, 120.0, 120.0]
    high = [100.0, 106.0, 120.0, 120.0, 120.0, 120.0]  # bar1 high 106 >= 105
    low = [100.0, 103.0, 119.0, 119.0, 119.0, 119.0]
    prices = _ohlcv(dates, close, high=high, low=low)
    spx = _flat_spx(dates)

    s = simulate_position_daily_alpha(
        prices, spx, entry_date=dates[0], tp=0.05, sl=-0.20, horizon_days=5
    )

    # Exit on the first lookahead bar => one daily return, to the TP price.
    assert list(s.index) == [dates[1]]
    np.testing.assert_allclose(s.values, [0.05], rtol=1e-12)


def test_tp_wins_tie_same_bar():
    """When TP and SL both trigger on the same bar, TP wins."""
    dates = BDAYS[:4]
    close = [100.0, 100.0, 100.0, 100.0]
    high = [100.0, 106.0, 106.0, 106.0]  # bar1 high 106 >= 105 (tp)
    low = [100.0, 94.0, 94.0, 94.0]  # bar1 low  94  <= 95  (sl)
    prices = _ohlcv(dates, close, high=high, low=low)
    spx = _flat_spx(dates)

    s = simulate_position_daily_alpha(
        prices, spx, entry_date=dates[0], tp=0.05, sl=-0.05, horizon_days=3
    )
    assert list(s.index) == [dates[1]]
    np.testing.assert_allclose(s.values, [0.05], rtol=1e-12)  # TP wins


def test_sl_hit():
    """SL hits on the second lookahead bar before TP."""
    dates = BDAYS[:6]
    close = [100.0, 99.0, 90.0, 90.0, 90.0, 90.0]
    high = [100.0, 100.0, 96.0, 96.0, 96.0, 96.0]  # never >= 105
    low = [100.0, 98.0, 89.0, 89.0, 89.0, 89.0]  # bar2 low 89 <= 90 (sl=-10%)
    prices = _ohlcv(dates, close, high=high, low=low)
    spx = _flat_spx(dates)

    s = simulate_position_daily_alpha(
        prices, spx, entry_date=dates[0], tp=0.50, sl=-0.10, horizon_days=5
    )

    # Two daily returns: bar1 close-to-close, bar2 to the SL threshold price (90).
    assert list(s.index) == [dates[1], dates[2]]
    expected = [99.0 / 100.0 - 1, 90.0 / 99.0 - 1]
    np.testing.assert_allclose(s.values, expected, rtol=1e-12)
    # Final compounded equals SL threshold return.
    np.testing.assert_allclose((1 + s.values).prod() - 1, -0.10, rtol=1e-12)


def test_horizon_exit_uses_close():
    """No TP/SL hit -> exit at horizon end on that day's Close."""
    dates = BDAYS[:5]
    close = [100.0, 101.0, 102.0, 103.0, 104.0]
    prices = _ohlcv(dates, close)
    spx = _flat_spx(dates)

    s = simulate_position_daily_alpha(
        prices, spx, entry_date=dates[0], tp=0.50, sl=-0.50, horizon_days=3
    )
    # horizon 3 -> lookahead bars 1,2,3 ; last index is dates[3]
    assert list(s.index) == [dates[1], dates[2], dates[3]]
    np.testing.assert_allclose((1 + s.values).prod() - 1, 103.0 / 100.0 - 1, rtol=1e-12)


def test_alpha_subtracts_spx_daily():
    """Alpha must subtract SPX close-to-close return each day."""
    dates = BDAYS[:4]
    close = [100.0, 110.0, 121.0, 121.0]
    spx_close = [200.0, 210.0, 210.0, 210.0]
    prices = _ohlcv(dates, close)
    spx = _spx_arrays(dates, spx_close)

    s = simulate_position_daily_alpha(
        prices, spx, entry_date=dates[0], tp=0.50, sl=-0.50, horizon_days=3
    )
    # Day1: stock 110/100-1=0.10, spx 210/200-1=0.05 -> alpha 0.05
    # Day2: stock 121/110-1=0.10, spx 210/210-1=0.0  -> alpha 0.10
    # Day3: stock 0.0, spx 0.0 -> 0.0
    expected = [0.10 - 0.05, 0.10 - 0.0, 0.0]
    np.testing.assert_allclose(s.values, expected, rtol=1e-12)


def test_missing_ohlcv_returns_empty():
    spx = _flat_spx(BDAYS[:5])
    assert simulate_position_daily_alpha(
        None, spx, entry_date=BDAYS[0], tp=0.05, sl=-0.05, horizon_days=5
    ).empty
    assert simulate_position_daily_alpha(
        pd.DataFrame(), spx, entry_date=BDAYS[0], tp=0.05, sl=-0.05, horizon_days=5
    ).empty


def test_entry_not_found_within_7_days_returns_empty():
    """First available trade date is >7 days after entry => empty."""
    dates = BDAYS[:5]
    prices = _ohlcv(dates, [100.0, 101, 102, 103, 104])
    spx = _flat_spx(dates)
    # entry_date 30 days before first bar -> tolerance exceeded
    early = dates[0] - pd.Timedelta(days=30)
    s = simulate_position_daily_alpha(
        prices, spx, entry_date=early, tp=0.05, sl=-0.05, horizon_days=3
    )
    assert s.empty


# ---------------------------------------------------------------------------
# simulate_daily_portfolio
# ---------------------------------------------------------------------------


def _make_loader(data: dict):
    """Return an injectable loader callable backed by a dict ticker->frame."""

    def loader(ticker, db_path_str=None, required_start_date=None):
        return data.get(ticker, pd.DataFrame())

    return loader


def test_portfolio_single_name_moves_book_by_weight_times_return():
    """A single open position at capital weight 0.05 moves the book by
    0.05 * its_return each day — NOT 100% (no renormalization)."""
    dates = BDAYS[:6]
    spx = _flat_spx(dates)
    a = 100.0 * np.cumprod(np.r_[1.0, np.full(5, 1.01)])  # +1%/day
    loader = _make_loader({"AAA": _ohlcv(dates, a)})

    pos = pd.DataFrame({"Ticker": ["AAA"], "entry_date": [dates[0]], "weight": [0.05]})
    df = simulate_daily_portfolio(
        pos, tp=0.50, sl=-0.50, horizon_days=5, ohlcv_loader=loader, spx_arrays=spx
    )
    # 0.05 capital weight * 0.01 daily return = 0.0005 per day (cash drag on rest).
    np.testing.assert_allclose(
        df["port_raw"].values, np.full(5, 0.05 * 0.01), rtol=1e-9
    )
    assert {"port_raw", "port_alpha"}.issubset(df.columns)


def test_portfolio_return_is_sum_of_weight_times_return():
    """Daily return == SUM(weight_i * ret_i) over open positions (absolute
    capital weights, not renormalized)."""
    dates = BDAYS[:6]
    spx = _flat_spx(dates)
    # AAA +1%/day, BBB -1%/day.
    a = 100.0 * np.cumprod(np.r_[1.0, np.full(5, 1.01)])
    b = 100.0 * np.cumprod(np.r_[1.0, np.full(5, 0.99)])
    data = {"AAA": _ohlcv(dates, a), "BBB": _ohlcv(dates, b)}
    loader = _make_loader(data)

    pos = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "entry_date": [dates[0], dates[0]],
            "weight": [0.04, 0.02],  # below the 5% cap, gross 0.06 < 1.0
        }
    )
    df = simulate_daily_portfolio(
        pos, tp=0.50, sl=-0.50, horizon_days=5, ohlcv_loader=loader, spx_arrays=spx
    )
    # 0.04 * 0.01 + 0.02 * (-0.01) = 0.0002 per day (NO renormalization).
    np.testing.assert_allclose(df["port_raw"].values, np.full(5, 0.0002), rtol=1e-9)


def test_portfolio_per_name_cap_applied():
    """Weights above per_name_cap are clamped to the cap before weighting."""
    dates = BDAYS[:6]
    spx = _flat_spx(dates)
    a = 100.0 * np.cumprod(np.r_[1.0, np.full(5, 1.01)])  # +1%/day
    loader = _make_loader({"AAA": _ohlcv(dates, a)})

    pos = pd.DataFrame({"Ticker": ["AAA"], "entry_date": [dates[0]], "weight": [0.50]})
    df = simulate_daily_portfolio(
        pos,
        tp=0.50,
        sl=-0.50,
        horizon_days=5,
        ohlcv_loader=loader,
        spx_arrays=spx,
        per_name_cap=0.05,
    )
    # Weight capped at 0.05 -> 0.05 * 0.01 = 0.0005 per day, not 0.50 * 0.01.
    np.testing.assert_allclose(df["port_raw"].values, np.full(5, 0.0005), rtol=1e-9)


def test_portfolio_gross_exposure_never_exceeds_cap():
    """When summed capped weights exceed max_gross_exposure, the day's weights
    scale down so total <= max_gross_exposure (per-name still <= cap)."""
    dates = BDAYS[:6]
    spx = _flat_spx(dates)
    # 30 names each +1%/day, capital weight 0.05 -> raw gross 1.50 > 1.0.
    n = 30
    close = 100.0 * np.cumprod(np.r_[1.0, np.full(5, 1.01)])
    tickers = [f"T{i:02d}" for i in range(n)]
    data = {t: _ohlcv(dates, close) for t in tickers}
    loader = _make_loader(data)

    pos = pd.DataFrame(
        {
            "Ticker": tickers,
            "entry_date": [dates[0]] * n,
            "weight": [0.05] * n,
        }
    )
    df = simulate_daily_portfolio(
        pos,
        tp=0.50,
        sl=-0.50,
        horizon_days=5,
        ohlcv_loader=loader,
        spx_arrays=spx,
        per_name_cap=0.05,
        max_gross_exposure=1.0,
    )
    # Gross scaled from 1.50 to 1.0; all names +1% -> portfolio == 1.0 * 0.01.
    np.testing.assert_allclose(df["port_raw"].values, np.full(5, 0.01), rtol=1e-9)
    # And it never exceeds the gross cap times the (uniform) daily return.
    assert (df["port_raw"].values <= 0.01 + 1e-12).all()


def test_portfolio_flat_on_days_with_no_open_position():
    """Days outside any position's holding window are flat (0 return)."""
    dates = BDAYS[:12]
    spx = _flat_spx(dates)
    # Position A: entry dates[0], horizon 2 -> open dates[1..2]
    # Position B: entry dates[6], horizon 2 -> open dates[7..8]
    close = 100.0 * np.cumprod(np.r_[1.0, np.full(11, 1.01)])
    data = {"AAA": _ohlcv(dates, close), "BBB": _ohlcv(dates, close)}
    loader = _make_loader(data)

    pos = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "entry_date": [dates[0], dates[6]],
            "weight": [1.0, 1.0],
        }
    )
    df = simulate_daily_portfolio(
        pos,
        tp=0.50,
        sl=-0.50,
        horizon_days=2,
        ohlcv_loader=loader,
        spx_arrays=spx,
    )
    # The gap day(s) between the two windows must be exactly 0.
    if "n_open" in df.columns:
        gap_days = df.index[(df.index > dates[2]) & (df.index < dates[7])]
        assert (df.loc[gap_days, "n_open"] == 0).all()
        assert (df.loc[gap_days, "port_raw"] == 0).all()


def test_portfolio_empty_positions_no_crash():
    spx = _flat_spx(BDAYS[:5])
    df = simulate_daily_portfolio(
        pd.DataFrame(columns=["Ticker", "entry_date", "weight"]),
        tp=0.05,
        sl=-0.05,
        horizon_days=5,
        ohlcv_loader=_make_loader({}),
        spx_arrays=spx,
    )
    assert df.empty
    # Metrics on an empty series must not crash.
    m = compute_portfolio_metrics(
        df["port_alpha"] if "port_alpha" in df else pd.Series(dtype=float)
    )
    assert np.isnan(m["sharpe"])
    assert m["n_days"] == 0


def test_portfolio_missing_ticker_skipped():
    """A ticker with no OHLCV is skipped; portfolio built from the rest."""
    dates = BDAYS[:6]
    spx = _flat_spx(dates)
    a = 100.0 * np.cumprod(np.r_[1.0, np.full(5, 1.01)])
    data = {"AAA": _ohlcv(dates, a)}  # BBB missing
    loader = _make_loader(data)

    pos = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "entry_date": [dates[0], dates[0]],
            "weight": [0.05, 0.05],
        }
    )
    df = simulate_daily_portfolio(
        pos,
        tp=0.50,
        sl=-0.50,
        horizon_days=5,
        ohlcv_loader=loader,
        spx_arrays=spx,
    )
    # Only AAA contributes; capital weight 0.05 * 1% -> 0.0005 per day.
    np.testing.assert_allclose(df["port_raw"].values, np.full(5, 0.0005), rtol=1e-9)


# ---------------------------------------------------------------------------
# entry_cost (optional column on positions)
# ---------------------------------------------------------------------------


def test_entry_cost_reduces_first_day_return_by_weight_times_cost():
    """A position's entry_cost lowers ONLY its first active day return, scaled
    by the position's capital weight: delta == weight * cost."""
    dates = BDAYS[:6]
    spx = _flat_spx(dates)
    a = 100.0 * np.cumprod(np.r_[1.0, np.full(5, 1.01)])  # +1%/day
    loader = _make_loader({"AAA": _ohlcv(dates, a)})

    weight, cost = 0.05, 0.004
    base = pd.DataFrame(
        {"Ticker": ["AAA"], "entry_date": [dates[0]], "weight": [weight]}
    )
    with_cost = base.assign(entry_cost=[cost])

    df_base = simulate_daily_portfolio(
        base, tp=0.50, sl=-0.50, horizon_days=5, ohlcv_loader=loader, spx_arrays=spx
    )
    df_cost = simulate_daily_portfolio(
        with_cost,
        tp=0.50,
        sl=-0.50,
        horizon_days=5,
        ohlcv_loader=loader,
        spx_arrays=spx,
    )

    # First active day reduced by weight * cost; later days unchanged.
    np.testing.assert_allclose(
        df_cost["port_raw"].iloc[0],
        df_base["port_raw"].iloc[0] - weight * cost,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        df_cost["port_alpha"].iloc[0],
        df_base["port_alpha"].iloc[0] - weight * cost,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        df_cost["port_raw"].iloc[1:].values,
        df_base["port_raw"].iloc[1:].values,
        rtol=1e-12,
    )


def test_entry_cost_hits_each_positions_own_first_day():
    """With staggered entries the cost lands on each position's own first day."""
    dates = BDAYS[:12]
    spx = _flat_spx(dates)
    close = 100.0 * np.cumprod(np.r_[1.0, np.full(11, 1.01)])
    loader = _make_loader({"AAA": _ohlcv(dates, close), "BBB": _ohlcv(dates, close)})

    pos = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "entry_date": [dates[0], dates[6]],
            "weight": [0.05, 0.05],
            "entry_cost": [0.01, 0.02],
        }
    )
    df = simulate_daily_portfolio(
        pos, tp=0.50, sl=-0.50, horizon_days=2, ohlcv_loader=loader, spx_arrays=spx
    )
    # AAA open dates[1..2]; first active day (dates[1]) = 0.05*(0.01 - 0.01) = 0.0.
    np.testing.assert_allclose(df.loc[dates[1], "port_raw"], 0.0, atol=1e-12)
    # BBB open dates[7..8]; first active day (dates[7]) = 0.05*(0.01 - 0.02) = -0.0005.
    np.testing.assert_allclose(df.loc[dates[7], "port_raw"], 0.05 * -0.01, atol=1e-12)


def test_no_entry_cost_column_is_backward_compatible():
    """Positions without an entry_cost column behave exactly as before."""
    dates = BDAYS[:6]
    spx = _flat_spx(dates)
    a = 100.0 * np.cumprod(np.r_[1.0, np.full(5, 1.01)])
    loader = _make_loader({"AAA": _ohlcv(dates, a)})
    pos = pd.DataFrame({"Ticker": ["AAA"], "entry_date": [dates[0]], "weight": [0.05]})
    df = simulate_daily_portfolio(
        pos, tp=0.50, sl=-0.50, horizon_days=5, ohlcv_loader=loader, spx_arrays=spx
    )
    # Capital weight 0.05 * 1% -> 0.0005 per day; no cost applied.
    np.testing.assert_allclose(df["port_raw"].values, np.full(5, 0.0005), rtol=1e-9)


# ---------------------------------------------------------------------------
# compute_portfolio_metrics
# ---------------------------------------------------------------------------


def test_metrics_match_hand_calc():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.03, -0.01])
    m = compute_portfolio_metrics(r)

    mean = r.mean()
    std = r.std(ddof=1)
    exp_sharpe = mean / std * np.sqrt(252)
    np.testing.assert_allclose(m["sharpe"], exp_sharpe, rtol=1e-12)
    np.testing.assert_allclose(m["ann_vol"], std * np.sqrt(252), rtol=1e-12)

    # Max drawdown on cumprod(1+r), positive magnitude.
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    exp_mdd = float(-dd.min())
    np.testing.assert_allclose(m["max_drawdown"], exp_mdd, rtol=1e-12)
    assert m["max_drawdown"] >= 0

    # CAGR from total compounded return over n/252 years.
    total = float(equity.iloc[-1]) - 1
    years = len(r) / 252.0
    exp_cagr = (1 + total) ** (1 / years) - 1
    np.testing.assert_allclose(m["cagr"], exp_cagr, rtol=1e-9)

    assert m["n_days"] == 6
    assert m["exposure_days"] == 5  # one zero day


def test_metrics_guard_short_and_constant_series():
    assert np.isnan(compute_portfolio_metrics(pd.Series([0.01]))["sharpe"])
    assert np.isnan(compute_portfolio_metrics(pd.Series(dtype=float))["sharpe"])
    # Constant non-zero returns -> std 0 -> sharpe nan, no crash.
    m = compute_portfolio_metrics(pd.Series([0.01, 0.01, 0.01]))
    assert np.isnan(m["sharpe"])


def test_metrics_sortino_uses_downside_only():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    m = compute_portfolio_metrics(r)
    downside = r[r < 0]
    dd_std = downside.std(ddof=1)
    exp_sortino = r.mean() / dd_std * np.sqrt(252)
    np.testing.assert_allclose(m["sortino"], exp_sortino, rtol=1e-9)


def test_realistic_sharpe_not_astronomical():
    """
    A genuine daily MTM curve from noisy prices must give a sane Sharpe,
    NOT the ~22 the old smoothed metric produced.
    """
    dates = BDAYS[:40]
    spx = _flat_spx(dates)
    rng = np.random.default_rng(123)
    # Slight positive drift + real daily noise.
    rets = rng.normal(0.0005, 0.02, len(dates))
    close = 100.0 * np.cumprod(np.r_[1.0, 1 + rets[1:]])
    data = {"AAA": _ohlcv(dates, close)}
    pos = pd.DataFrame({"Ticker": ["AAA"], "entry_date": [dates[0]], "weight": [1.0]})
    df = simulate_daily_portfolio(
        pos,
        tp=5.0,
        sl=-5.0,
        horizon_days=38,
        ohlcv_loader=_make_loader(data),
        spx_arrays=spx,
    )
    m = compute_portfolio_metrics(df["port_alpha"])
    assert abs(m["sharpe"]) < 10  # not astronomical
