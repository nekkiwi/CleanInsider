# tests/test_validated_strategy_filter.py
"""
Tests for the LIVE validated-strategy filter (liquid + CEO/CFO gate) and the
get_adv helper that backs the live liquidity gate.

The validated config that must trade live:
  - LightGBM ensemble, strategy 1w_tp0p05_sl-0p05
  - LIQUID names only: live Price >= LIQUID_PRICE_MIN AND live ADV >= LIQUID_ADV_MIN
  - CEO/CFO insider buys only (CEO==1 OR CFO==1)
  - missing ADV => dropped (unknown liquidity = untradeable)
  - gated by config.LIVE_LIQUID_CEOCFO_FILTER (default True) so it is reversible

No live API calls: the AlpacaTradingClient is constructed via __new__ with a
mocked data_client, mirroring tests/test_sizing.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_inference import apply_validated_strategy_filter  # noqa: E402
from src import config  # noqa: E402
from src.alpaca.trading_client import AlpacaTradingClient  # noqa: E402


# --------------------------------------------------------------------------- #
# get_adv
# --------------------------------------------------------------------------- #
def _make_bars(closes, volumes):
    """Build a list of objects mimicking alpaca-py Bar attributes."""
    bars = []
    for c, v in zip(closes, volumes):
        bar = MagicMock()
        bar.close = c
        bar.volume = v
        bars.append(bar)
    return bars


class TestGetADV:
    def _client_with_data(self, data_map):
        """An AlpacaTradingClient with client/data_client injected (no network)."""
        client = AlpacaTradingClient.__new__(AlpacaTradingClient)
        client.client = MagicMock()  # marks "connected"
        data_client = MagicMock()

        barset = MagicMock()
        barset.data = data_map  # {symbol: [bar, ...]}
        data_client.get_stock_bars.return_value = barset

        client.data_client = data_client
        return client

    def test_adv_median_dollar_volume(self):
        # Constant close 10, constant volume 1_000_000 -> dollar vol 10_000_000.
        # Median of constant series == 10_000_000.
        closes = [10.0] * 60
        volumes = [1_000_000] * 60
        client = self._client_with_data({"AAA": _make_bars(closes, volumes)})

        adv = client.get_adv(["AAA"], period=60)
        assert pytest.approx(adv["AAA"], rel=1e-9) == 10_000_000.0

    def test_adv_uses_median_not_mean(self):
        # One huge volume spike; MEDIAN ignores it, a MEAN would not.
        closes = [10.0] * 60
        volumes = [1_000_000] * 60
        volumes[30] = 1_000_000_000  # extreme outlier day
        client = self._client_with_data({"AAA": _make_bars(closes, volumes)})

        adv = client.get_adv(["AAA"], period=60)
        assert pytest.approx(adv["AAA"], rel=1e-9) == 10_000_000.0

    def test_adv_missing_symbol_omitted(self):
        client = self._client_with_data({})  # no data for anything
        adv = client.get_adv(["MISSING"], period=60)
        assert "MISSING" not in adv
        assert adv == {}

    def test_adv_empty_bars_omitted(self):
        client = self._client_with_data({"CCC": []})
        adv = client.get_adv(["CCC"], period=60)
        assert "CCC" not in adv

    def test_adv_no_client_returns_empty(self):
        client = AlpacaTradingClient.__new__(AlpacaTradingClient)
        client.client = None
        adv = client.get_adv(["AAA"])
        assert adv == {}


# --------------------------------------------------------------------------- #
# apply_validated_strategy_filter
# --------------------------------------------------------------------------- #
class _FakeClient:
    """Minimal stand-in for AlpacaTradingClient exposing get_adv only."""

    def __init__(self, adv_map):
        self._adv_map = adv_map

    def get_adv(self, symbols, period=60):
        return {s: self._adv_map[s] for s in symbols if s in self._adv_map}


def _signals_df():
    """5 signals spanning every gate outcome.

    GOOD  : CEO buy, $50, ADV $50M           -> KEEP
    GOODF : CFO buy, $25, ADV $10M           -> KEEP
    CHEAP : CEO buy, $5 (< $10), ADV $50M    -> DROP (price)
    THIN  : CFO buy, $40, ADV $1M (< $5M)    -> DROP (low adv)
    NOTC  : non-CEO/CFO, $80, ADV $100M      -> DROP (not CEO/CFO)
    NOADV : CEO buy, $30, ADV missing        -> DROP (unknown liquidity)
    """
    return pd.DataFrame(
        {
            "Ticker": ["GOOD", "GOODF", "CHEAP", "THIN", "NOTC", "NOADV"],
            "Filing Date": pd.to_datetime(["2024-01-02"] * 6),
            "buy_signal": [1, 1, 1, 1, 1, 1],
            "predicted_return": [0.10, 0.08, 0.20, 0.07, 0.12, 0.09],
            "confidence": [0.8, 0.7, 0.9, 0.6, 0.75, 0.7],
            "Price": [50.0, 25.0, 5.0, 40.0, 80.0, 30.0],
            "CEO": [1, 0, 1, 0, 0, 1],
            "CFO": [0, 1, 0, 1, 0, 0],
        }
    )


def _adv_map():
    return {
        "GOOD": 50e6,
        "GOODF": 10e6,
        "CHEAP": 50e6,
        "THIN": 1e6,
        "NOTC": 100e6,
        # NOADV intentionally absent -> unknown liquidity
    }


def test_filter_keeps_only_liquid_ceocfo(monkeypatch):
    monkeypatch.setattr(config, "LIVE_LIQUID_CEOCFO_FILTER", True)
    monkeypatch.setattr(config, "LIQUID_PRICE_MIN", 10.0)
    monkeypatch.setattr(config, "LIQUID_ADV_MIN", 5_000_000.0)

    out = apply_validated_strategy_filter(_signals_df(), _FakeClient(_adv_map()))

    kept = set(out["Ticker"])
    assert kept == {"GOOD", "GOODF"}
    # adv column surfaced for the survivors
    assert "adv" in out.columns
    assert (out["adv"] >= config.LIQUID_ADV_MIN).all()
    assert (out["Price"] >= config.LIQUID_PRICE_MIN).all()


def test_filter_drops_subprice(monkeypatch):
    monkeypatch.setattr(config, "LIVE_LIQUID_CEOCFO_FILTER", True)
    monkeypatch.setattr(config, "LIQUID_PRICE_MIN", 10.0)
    monkeypatch.setattr(config, "LIQUID_ADV_MIN", 5_000_000.0)

    out = apply_validated_strategy_filter(_signals_df(), _FakeClient(_adv_map()))
    assert "CHEAP" not in set(out["Ticker"])


def test_filter_drops_low_adv(monkeypatch):
    monkeypatch.setattr(config, "LIVE_LIQUID_CEOCFO_FILTER", True)
    monkeypatch.setattr(config, "LIQUID_PRICE_MIN", 10.0)
    monkeypatch.setattr(config, "LIQUID_ADV_MIN", 5_000_000.0)

    out = apply_validated_strategy_filter(_signals_df(), _FakeClient(_adv_map()))
    assert "THIN" not in set(out["Ticker"])


def test_filter_drops_missing_adv(monkeypatch):
    monkeypatch.setattr(config, "LIVE_LIQUID_CEOCFO_FILTER", True)
    monkeypatch.setattr(config, "LIQUID_PRICE_MIN", 10.0)
    monkeypatch.setattr(config, "LIQUID_ADV_MIN", 5_000_000.0)

    out = apply_validated_strategy_filter(_signals_df(), _FakeClient(_adv_map()))
    assert "NOADV" not in set(out["Ticker"])


def test_filter_drops_non_ceocfo(monkeypatch):
    monkeypatch.setattr(config, "LIVE_LIQUID_CEOCFO_FILTER", True)
    monkeypatch.setattr(config, "LIQUID_PRICE_MIN", 10.0)
    monkeypatch.setattr(config, "LIQUID_ADV_MIN", 5_000_000.0)

    out = apply_validated_strategy_filter(_signals_df(), _FakeClient(_adv_map()))
    assert "NOTC" not in set(out["Ticker"])


def test_filter_disabled_passthrough(monkeypatch):
    # Flag off -> no filtering, frame returned unchanged (no get_adv needed).
    monkeypatch.setattr(config, "LIVE_LIQUID_CEOCFO_FILTER", False)

    df = _signals_df()
    out = apply_validated_strategy_filter(df, _FakeClient({}))
    assert len(out) == len(df)
    assert set(out["Ticker"]) == set(df["Ticker"])


def test_filter_empty_input(monkeypatch):
    monkeypatch.setattr(config, "LIVE_LIQUID_CEOCFO_FILTER", True)
    out = apply_validated_strategy_filter(pd.DataFrame(), _FakeClient({}))
    assert out.empty


def test_filter_missing_ceocfo_columns_keeps_liquid(monkeypatch):
    # If CEO/CFO are not surfaced, the CEO/CFO gate is skipped (logged) but the
    # liquid gate still applies -- documents the live_features follow-up.
    monkeypatch.setattr(config, "LIVE_LIQUID_CEOCFO_FILTER", True)
    monkeypatch.setattr(config, "LIQUID_PRICE_MIN", 10.0)
    monkeypatch.setattr(config, "LIQUID_ADV_MIN", 5_000_000.0)

    df = _signals_df().drop(columns=["CEO", "CFO"])
    out = apply_validated_strategy_filter(df, _FakeClient(_adv_map()))
    # No CEO/CFO gate -> liquid survivors are GOOD, GOODF, NOTC ($80/$100M ok).
    assert set(out["Ticker"]) == {"GOOD", "GOODF", "NOTC"}


def test_sleeve_scale_default_is_one():
    assert hasattr(config, "SLEEVE_SCALE")
    assert config.SLEEVE_SCALE == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
