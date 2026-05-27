"""Unit tests for src/scrapers/feature_scraper/generate_adv.py.

Verify the point-in-time ADV computation on synthetic OHLCV:
  - 60-day rolling MEDIAN of Close*Volume (min_periods=20),
  - taken AS-OF the filing date (last rolling value with date <= Filing Date),
  - NO look-ahead (a filing before the min_periods window or before any data is NaN),
  - per-ticker isolation (one ticker's prices never contaminate another).
The OHLCV loader is injected so nothing touches disk.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapers.feature_scraper.generate_adv import (  # noqa: E402
    compute_pit_adv_for_ticker,
    generate_adv,
)

BDAYS = pd.bdate_range("2024-01-01", periods=120)


def _ohlcv(dates, close, volume):
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_pit_adv_picks_last_value_on_or_before_filing_date():
    # Constant price 10, constant volume 1_000_000 => dollar vol = 10_000_000.
    # 60-day rolling median == 10_000_000 once >=20 obs exist.
    dates = BDAYS[:80]
    ohlcv = _ohlcv(dates, close=[10.0] * 80, volume=[1_000_000] * 80)

    # Filing on a trading day well past the warmup window.
    filing = dates[50]
    adv = compute_pit_adv_for_ticker(ohlcv, [filing])
    assert len(adv) == 1
    assert np.isclose(adv[0], 10_000_000.0)


def test_pit_adv_asof_uses_last_value_before_nontrading_filing():
    # Filing date falls on a weekend/gap; asof must pick the last trading day <= it.
    dates = BDAYS[:80]
    # Increasing dollar volume so the asof choice is observable: vol grows daily.
    close = np.full(80, 10.0)
    volume = np.arange(1, 81) * 100_000.0
    ohlcv = _ohlcv(dates, close, volume)

    # Pick a Friday so the next two calendar days are the weekend (no bars).
    fridays = [d for d in dates if d.weekday() == 4 and d >= dates[30]]
    last_trading = fridays[0]
    filing = last_trading + pd.Timedelta(days=2)  # Sunday: weekend gap, no bar
    assert filing.weekday() == 6  # sanity: it is a Sunday
    adv_gap = compute_pit_adv_for_ticker(ohlcv, [filing])
    adv_on = compute_pit_adv_for_ticker(ohlcv, [last_trading])
    # asof on a gap day == value at the last trading day on/before it.
    assert np.isclose(adv_gap[0], adv_on[0])


def test_pit_adv_no_lookahead_before_min_periods():
    # Fewer than min_periods (20) observations on/before the filing date => NaN.
    dates = BDAYS[:80]
    ohlcv = _ohlcv(dates, close=[10.0] * 80, volume=[1_000_000] * 80)
    early_filing = dates[5]  # only 6 obs precede it (< 20)
    adv = compute_pit_adv_for_ticker(ohlcv, [early_filing])
    assert np.isnan(adv[0])


def test_pit_adv_before_any_data_is_nan():
    dates = BDAYS[10:80]
    ohlcv = _ohlcv(dates, close=[10.0] * 70, volume=[1_000_000] * 70)
    filing_before = BDAYS[0]  # earlier than every bar
    adv = compute_pit_adv_for_ticker(ohlcv, [filing_before])
    assert np.isnan(adv[0])


def test_pit_adv_uses_median_not_mean():
    # Inject one huge volume spike; the 60-day MEDIAN ignores it, a MEAN would not.
    dates = BDAYS[:80]
    close = np.full(80, 10.0)
    volume = np.full(80, 1_000_000.0)
    volume[40] = 1_000_000_000.0  # one extreme outlier day
    ohlcv = _ohlcv(dates, close, volume)
    filing = dates[70]
    adv = compute_pit_adv_for_ticker(ohlcv, [filing])
    # Median dollar vol stays at the baseline 10_000_000 despite the spike.
    assert np.isclose(adv[0], 10_000_000.0)


def test_generate_adv_end_to_end_with_injected_loader(tmp_path):
    # Two tickers with distinct dollar volumes; verify per-ticker isolation
    # and the output schema/path.
    dates = BDAYS[:80]
    data = {
        "AAA": _ohlcv(dates, close=[10.0] * 80, volume=[1_000_000] * 80),
        "BBB": _ohlcv(dates, close=[20.0] * 80, volume=[2_000_000] * 80),
    }

    def loader(ticker, db_path_str=None, local_only=None):
        return data.get(ticker, pd.DataFrame())

    master = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "ZZZ"],
            "Filing Date": [dates[50], dates[50], dates[50]],
            "Price": [10.0, 20.0, 1.0],
        }
    )
    out_path = tmp_path / "adv.parquet"
    result = generate_adv(
        master_df=master,
        db_path_str="ignored",
        loader=loader,
        output_path=out_path,
    )

    assert out_path.exists()
    assert list(result.columns) == ["Ticker", "Filing Date", "adv"]
    by_ticker = result.set_index("Ticker")["adv"]
    assert np.isclose(by_ticker["AAA"], 10_000_000.0)
    assert np.isclose(by_ticker["BBB"], 40_000_000.0)
    # ZZZ has no OHLCV -> adv is NaN (unknown liquidity).
    assert np.isnan(by_ticker["ZZZ"])
